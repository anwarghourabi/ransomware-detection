"""Tâche 6 : séquences par exécution, préfixes 10/25/50/75/100 %, découpages figés.

Entrée  : table de features par fenêtre (tâche 5), une ligne par (exécution, fenêtre).
Sortie  : X (exécutions, T_MAX, features) + longueurs + étiquettes + table des découpages.
Tout est en pandas / numpy : la table de features est petite (~400 000 lignes).
"""
import json
import math

import numpy as np
import pandas as pd

META = ["rel", "category", "cpu", "ram", "family", "sample", "label"]
FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
MIN_WINDOWS = 100          # exécutions plus courtes = traces incomplètes, écartées
SEED = 42
# features à queue lourde (comptes, octets, adresses) : passées en log(1 + x)
LOG_SUFFIXES = ("_n", "_bytes", "_addr_uniq", "_addr_std")


# ---------- 1. nettoyage ----------
def keep_runs(fe: pd.DataFrame, min_windows=MIN_WINDOWS):
    n = fe.groupby("rel")["w"].max() + 1
    short = n[n < min_windows]
    return fe[~fe.rel.isin(short.index)].copy(), short


def useful_features(fe: pd.DataFrame, feats) -> list:
    """Retire les features constantes (ex. entropie des lectures, jamais mesurée)."""
    return [c for c in feats if fe[c].nunique() > 1]


# ---------- 2. séquences ----------
def to_sequences(fe: pd.DataFrame, feats, t_max=None):
    """Grille complète 0..n-1 par exécution (fenêtres vides = 0), puis tableau 3D.
    Renvoie X (N, T_MAX, F) float32, lengths (N,), runs (N lignes de méta, même ordre)."""
    fe = fe.sort_values(["rel", "w"])
    runs = fe.groupby("rel", sort=True)[META[1:]].first().reset_index()
    runs["n_windows"] = (fe.groupby("rel", sort=True)["w"].max() + 1).values
    t_max = t_max or int(runs.n_windows.max())
    X = np.zeros((len(runs), t_max, len(feats)), dtype=np.float32)
    idx = {r: i for i, r in enumerate(runs.rel)}
    w = fe["w"].to_numpy()
    keep = w < t_max
    rows = fe["rel"].map(idx).to_numpy()
    X[rows[keep], w[keep]] = fe.loc[keep, feats].to_numpy(dtype=np.float32)
    lengths = np.minimum(runs.n_windows.to_numpy(), t_max).astype(np.int32)
    return X, lengths, runs


def log_transform(X, feats):
    cols = [i for i, f in enumerate(feats) if f.endswith(LOG_SUFFIXES)]
    X[:, :, cols] = np.log1p(np.clip(X[:, :, cols], 0, None))
    return X


def fit_scaler(X, lengths, train_mask):
    """Moyenne / écart-type par feature, calculés sur les fenêtres réelles du TRAIN seulement."""
    valid = np.arange(X.shape[1])[None, :] < lengths[:, None]
    sel = valid & train_mask[:, None]
    vals = X[sel]
    mean, std = vals.mean(0), vals.std(0)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_scaler(X, lengths, mean, std):
    valid = np.arange(X.shape[1])[None, :] < lengths[:, None]
    X = (X - mean) / std
    X[~valid] = 0.0           # padding remis à 0 après normalisation
    return X.astype(np.float32)


# ---------- 3. préfixes ----------
def prefix_lengths(lengths, frac):
    """Nombre de fenêtres observées quand on voit `frac` de chaque exécution (au moins 1)."""
    return np.maximum(1, np.ceil(lengths * frac - 1e-9)).astype(np.int32)


def prefix(X, lengths, frac):
    """Copie de X où tout ce qui est après le préfixe est mis à 0. Renvoie (X_p, longueurs_p)."""
    lp = prefix_lengths(lengths, frac)
    Xp = X.copy()
    Xp[np.arange(X.shape[1])[None, :] >= lp[:, None]] = 0.0
    return Xp, lp


# ---------- 4. découpages figés (par exécution, jamais par fenêtre) ----------
def make_splits(runs: pd.DataFrame, val=0.1, test=0.2, seed=SEED) -> pd.Series:
    """original -> train / val / test, stratifié par (programme, configuration) ;
    extra -> test_hw (matériel jamais vu) ; mix -> test_mix ; variants -> test_variants."""
    rng = np.random.default_rng(seed)
    split = pd.Series("", index=runs.index, dtype=object)
    other = {"extra": "test_hw", "mix": "test_mix", "variants": "test_variants"}
    for cat, name in other.items():
        split[runs.category == cat] = name
    orig = runs[runs.category == "original"]
    for _, g in orig.groupby(["sample", "cpu", "ram"]):
        ids = rng.permutation(g.index.to_numpy())
        n_test = max(1, round(len(ids) * test))
        n_val = max(1, round(len(ids) * val)) if len(ids) >= 3 else 0
        split[ids[:n_test]] = "test"
        split[ids[n_test:n_test + n_val]] = "val"
        split[ids[n_test + n_val:]] = "train"
    return split


def lofo_folds(runs: pd.DataFrame):
    """RQ3 : une famille de ransomware retirée à chaque tour (variants Conti = Conti).
    train = original/train sans la famille ; test = la famille (toutes catégories) + bénins de test."""
    fams = sorted(runs.loc[(runs.label == 1) & (runs.category == "original"), "family"].unique())
    folds = {}
    for f in fams:
        tr = (runs.split == "train") & (runs.family != f)
        te = (runs.family == f) | ((runs.label == 0) & (runs.split == "test"))
        folds[f] = (np.where(tr)[0], np.where(te)[0])
    return folds


# ---------- 5. sauvegarde ----------
def save(out_dir, X, lengths, runs, feats, mean, std):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "sequences.npz", X=X, lengths=lengths,
                        labels=runs.label.to_numpy(np.int8))
    runs.to_csv(out_dir / "runs.csv", index=False)
    (out_dir / "preprocessing.json").write_text(json.dumps(
        {"features": feats, "mean": mean.tolist(), "std": std.tolist(),
         "log_features": [f for f in feats if f.endswith(LOG_SUFFIXES)],
         "min_windows": MIN_WINDOWS, "seed": SEED, "fractions": FRACTIONS}, indent=1))


def load(out_dir):
    """Pour Ghada / les modèles : X, longueurs, étiquettes, table des exécutions (avec split)."""
    d = np.load(out_dir / "sequences.npz")
    return d["X"], d["lengths"], d["labels"], pd.read_csv(out_dir / "runs.csv")
