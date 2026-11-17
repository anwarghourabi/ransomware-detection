"""Tâche 9 : modèles séquentiels CNN 1D, LSTM, GRU (PyTorch).

Entrée : X (exécutions, fenêtres, features) déjà normalisé, longueurs réelles, étiquettes (tâche 6).
Les séquences sont remplies de 0 à droite : les modèles utilisent les longueurs pour ignorer ce padding.
Les LSTM / GRU sont unidirectionnels (lecture du passé vers le présent), comme pour une détection en ligne.
Tâche 10 : MHSA-LSTM = LSTM + attention multi-têtes causale, avec option de pondération des
features par gain d'information (Alqahtani et al. 2026).
"""
import copy
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ---------- 1. modèles ----------
class CNN1D(nn.Module):
    """Convolutions sur le temps, puis moyenne et max sur les fenêtres réelles.
    Le padding est remis à 0 avant chaque convolution pour que la fin d'une séquence
    courte ne « voie » jamais les fenêtres de remplissage."""
    def __init__(self, n_feat, hidden=64, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(n_feat, hidden, kernel_size=5, padding=2),
                                   nn.ReLU(), nn.BatchNorm1d(hidden))
        self.conv2 = nn.Sequential(nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
                                   nn.ReLU(), nn.BatchNorm1d(hidden))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 1))

    def forward(self, x, lengths):
        mask = (torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]).unsqueeze(1)
        h = self.conv1(x.transpose(1, 2) * mask)                     # (B, H, T)
        h = self.conv2(h * mask) * mask
        mean = h.sum(2) / lengths[:, None].clamp(min=1)
        mx = h.masked_fill(~mask, -1e9).max(2).values
        return self.head(torch.cat([mean, mx], 1)).squeeze(1)


class RNN(nn.Module):
    """LSTM ou GRU : l'état caché après la dernière fenêtre réelle résume la séquence."""
    def __init__(self, n_feat, kind="lstm", hidden=64, layers=1, dropout=0.3):
        super().__init__()
        cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = cls(n_feat, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        h = h[0] if isinstance(h, tuple) else h                     # LSTM renvoie (h, c)
        return self.head(h[-1]).squeeze(1)


class AttnLSTM(nn.Module):
    """MHSA-LSTM : LSTM, puis attention multi-têtes sur ses sorties.
    - attention causale : la fenêtre t ne regarde que les fenêtres 1..t (pas le futur) ;
    - le padding est masqué ;
    - la prédiction vient de la dernière fenêtre réelle, qui « regarde » tout le passé.
    feat_weights (optionnel) : un poids par feature (gain d'information), multiplié à l'entrée."""
    def __init__(self, n_feat, hidden=64, heads=4, dropout=0.3, feat_weights=None):
        super().__init__()
        w = torch.ones(n_feat) if feat_weights is None else torch.as_tensor(feat_weights, dtype=torch.float32)
        self.register_buffer("feat_w", w)
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.last_attn = None                                        # (B, T) : pour les figures

    def forward(self, x, lengths):
        T = x.size(1)
        x = x * self.feat_w
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=T)
        pad = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]          # True = padding
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)       # True = futur interdit
        a, w = self.attn(h, h, h, key_padding_mask=pad, attn_mask=causal, need_weights=True)
        z = self.norm(h + a)
        last = (lengths - 1).clamp(min=0)
        rows = torch.arange(x.size(0), device=x.device)
        self.last_attn = w[rows, last].detach()                     # moyenne des têtes
        return self.head(z[rows, last]).squeeze(1)


def make_model(name, n_feat, **kw):
    if name == "CNN":
        return CNN1D(n_feat, **kw)
    if name in ("LSTM", "GRU"):
        return RNN(n_feat, kind=name.lower(), **kw)
    if name.startswith("MHSA-LSTM"):
        return AttnLSTM(n_feat, **kw)
    raise ValueError(name)


def info_gain(X, lengths, y, idx, bins=10):
    """Gain d'information de chaque feature (calculé sur le train seulement).
    Chaque exécution est résumée par la moyenne de la feature sur ses fenêtres réelles,
    découpée en `bins` classes (quantiles) ; IG = H(y) - H(y | classe)."""
    idx = np.asarray(idx)
    mask = np.arange(X.shape[1])[None, :] < lengths[idx, None]
    m = (X[idx] * mask[..., None]).sum(1) / lengths[idx, None]
    yy = np.asarray(y)[idx].astype(int)

    def H(v):
        p = np.bincount(v, minlength=2) / max(len(v), 1)
        p = p[p > 0]
        return -(p * np.log2(p)).sum()

    ig = []
    for j in range(m.shape[1]):
        edges = np.unique(np.quantile(m[:, j], np.linspace(0, 1, bins + 1)[1:-1]))
        c = np.digitize(m[:, j], edges)
        ig.append(H(yy) - sum((c == k).mean() * H(yy[c == k]) for k in np.unique(c)))
    return np.array(ig)


def ig_weights(ig):
    """Poids d'entrée : IG normalisé pour que la moyenne vaille 1 (une feature sans IG reste à 0.1)."""
    w = np.asarray(ig) / max(np.mean(ig), 1e-12)
    return np.clip(w, 0.1, None).astype(np.float32)


def attention_profile(model, X, lengths, idx, n_bins=20, batch_size=256):
    """Où regarde le modèle ? Attention de la dernière fenêtre, ramenée à 20 tranches de 5 %
    de la séquence (0 = début, 1 = fin). Renvoie une matrice (exécutions, n_bins)."""
    model.eval()
    out = []
    with torch.no_grad():
        for b in _batches(np.asarray(idx), batch_size, False, None):
            L = int(lengths[b].max())
            model(torch.from_numpy(X[b, :L]), torch.from_numpy(lengths[b]).long())
            w = model.last_attn.numpy()
            for i, n in enumerate(lengths[b]):
                pos = np.minimum((np.arange(n) / n * n_bins).astype(int), n_bins - 1)
                out.append(np.bincount(pos, weights=w[i, :n], minlength=n_bins))
    return np.array(out)


# ---------- 2. entraînement ----------
def _batches(idx, batch_size, shuffle, rng):
    idx = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(idx), batch_size):
        yield idx[i:i + batch_size]


def predict_proba(model, X, lengths, idx, batch_size=256):
    model.eval()
    out = []
    with torch.no_grad():
        for b in _batches(np.asarray(idx), batch_size, False, None):
            L = int(lengths[b].max())
            xb = torch.from_numpy(X[b, :L]); lb = torch.from_numpy(lengths[b]).long()
            out.append(torch.sigmoid(model(xb, lb)).numpy())
    return np.concatenate(out) if out else np.array([])


def _prefix_len(lengths, frac):
    return np.maximum(1, np.ceil(lengths * frac - 1e-9)).astype(lengths.dtype)


def val_loss(model, X, lengths, y, val_idx, prefix_fracs=None):
    """Perte BCE sur la validation ; avec prefix_fracs, moyenne sur ces fractions."""
    yv = torch.from_numpy(y[val_idx]).float()
    out = []
    for f in (prefix_fracs or [1.0]):
        lp = _prefix_len(lengths, f)
        pv = torch.from_numpy(predict_proba(model, X, lp, val_idx)).clamp(1e-6, 1 - 1e-6)
        out.append(nn.functional.binary_cross_entropy(pv, yv).item())
    return float(np.mean(out))


def train(name, X, lengths, y, train_idx, val_idx, seed=42, epochs=40, patience=6,
          batch_size=64, lr=1e-3, weight_decay=1e-4, verbose=True, prefix_fracs=None, **model_kw):
    """Entraîne avec arrêt précoce sur la perte de validation ; renvoie le meilleur modèle.
    - modèle « par fraction » : passer directement les longueurs du préfixe dans `lengths`
      (les fenêtres après le préfixe sont ignorées par les modèles) ;
    - modèle « multi-préfixes » : prefix_fracs=[0.1, 0.25, ...] ; à chaque batch, chaque
      exécution est coupée à une fraction tirée au hasard dans cette liste."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(name, X.shape[2], **model_kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_loss, wait = None, np.inf, 0
    for ep in range(epochs):
        t = time.time(); model.train(); tot = 0.0
        for b in _batches(np.asarray(train_idx), batch_size, True, rng):
            lb = lengths[b]
            if prefix_fracs:
                lb = _prefix_len(lb, rng.choice(prefix_fracs, size=len(b)))
            L = int(lb.max())
            xb = torch.from_numpy(X[b, :L]); lb = torch.from_numpy(lb).long()
            yb = torch.from_numpy(y[b]).float()
            opt.zero_grad()
            loss = loss_fn(model(xb, lb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        vloss = val_loss(model, X, lengths, y, val_idx, prefix_fracs)
        if verbose:
            print(f"  {name} ép. {ep + 1:2d}  perte train {tot / len(train_idx):.4f}  "
                  f"perte val {vloss:.4f}  ({time.time() - t:.1f} s)")
        if vloss < best_loss - 1e-4:
            best, best_loss, wait = copy.deepcopy(model.state_dict()), vloss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best)
    return model
