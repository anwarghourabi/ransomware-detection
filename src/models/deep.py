"""Tâche 9 : modèles séquentiels CNN 1D, LSTM, GRU (PyTorch).

Entrée : X (exécutions, fenêtres, features) déjà normalisé, longueurs réelles, étiquettes (tâche 6).
Les séquences sont remplies de 0 à droite : les modèles utilisent les longueurs pour ignorer ce padding.
Les LSTM / GRU sont unidirectionnels (lecture du passé vers le présent), comme pour une détection en ligne.
"""
import copy
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


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


def make_model(name, n_feat, **kw):
    if name == "CNN":
        return CNN1D(n_feat, **kw)
    if name in ("LSTM", "GRU"):
        return RNN(n_feat, kind=name.lower(), **kw)
    raise ValueError(name)


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


def train(name, X, lengths, y, train_idx, val_idx, seed=42, epochs=40, patience=6,
          batch_size=64, lr=1e-3, weight_decay=1e-4, verbose=True, **model_kw):
    """Entraîne avec arrêt précoce sur la perte de validation ; renvoie le meilleur modèle."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(name, X.shape[2], **model_kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_loss, wait = None, np.inf, 0
    yv = torch.from_numpy(y[val_idx]).float()
    for ep in range(epochs):
        t = time.time(); model.train(); tot = 0.0
        for b in _batches(np.asarray(train_idx), batch_size, True, rng):
            L = int(lengths[b].max())
            xb = torch.from_numpy(X[b, :L]); lb = torch.from_numpy(lengths[b]).long()
            yb = torch.from_numpy(y[b]).float()
            opt.zero_grad()
            loss = loss_fn(model(xb, lb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        pv = torch.from_numpy(predict_proba(model, X, lengths, val_idx)).clamp(1e-6, 1 - 1e-6)
        vloss = nn.functional.binary_cross_entropy(pv, yv).item()
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
