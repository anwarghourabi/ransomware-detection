"""Tâche 12 (RQ2) : courbes ROC / PR et détection en ligne (score glissant + seuil).

Détection en ligne (comme RansoGuard, Cen et al. 2024) : toutes les `step` secondes, le modèle
donne un score sur tout ce qu'il a vu depuis le début de l'exécution. L'alerte est levée à la
première vérification où le score dépasse le seuil (k fois de suite si k > 1).
"""
import numpy as np

from src.models.deep import predict_proba


# ---------- 1. courbes ROC et PR (numpy) ----------
def roc_curve(y, p):
    """Renvoie (fpr, tpr) pour tous les seuils, du plus strict au plus permissif."""
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    o = np.argsort(-p, kind="mergesort")
    y, p = y[o], p[o]
    keep = np.r_[np.where(np.diff(p))[0], len(p) - 1]           # un point par seuil distinct
    tp = np.cumsum(y)[keep]; fp = np.cumsum(1 - y)[keep]
    tpr = np.r_[0, tp / max(y.sum(), 1)]; fpr = np.r_[0, fp / max((1 - y).sum(), 1)]
    return fpr, tpr


def pr_curve(y, p):
    """Renvoie (rappel, précision) pour tous les seuils."""
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    o = np.argsort(-p, kind="mergesort")
    y, p = y[o], p[o]
    keep = np.r_[np.where(np.diff(p))[0], len(p) - 1]
    tp = np.cumsum(y)[keep]; fp = np.cumsum(1 - y)[keep]
    rec = np.r_[0, tp / max(y.sum(), 1)]; prec = np.r_[1, tp / np.maximum(tp + fp, 1)]
    return rec, prec


def auc(x, y):
    """Aire sous une courbe (trapèzes), x croissant."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


# ---------- 2. scores en ligne ----------
def online_scores(model, X, lengths, idx, step=5, t_max=None):
    """Score du modèle après t = step, 2·step, … fenêtres (secondes) observées.
    Renvoie (T, S) : T = instants (s), S = scores (exécutions, instants), NaN après la fin du run."""
    idx = np.asarray(idx)
    L = lengths[idx]
    t_max = int(t_max or L.max())
    T = np.arange(step, t_max + step, step)
    S = np.full((len(idx), len(T)), np.nan, dtype=np.float32)
    for j, t in enumerate(T):
        # runs encore en cours à t, et runs qui se terminent entre t - step et t (dernière vérification)
        use = L > t - step
        if not use.any():
            continue
        tmp = lengths.copy(); tmp[idx[use]] = np.minimum(L[use], t)
        S[use, j] = predict_proba(model, X, tmp, idx[use])
    return T, S


def first_alarm(T, S, thr, k=1):
    """Instant (s) de la première alerte : k vérifications consécutives avec score >= thr.
    NaN si aucune alerte avant la fin du run."""
    hit = np.nan_to_num(S, nan=-1.0) >= thr
    out = np.full(len(S), np.nan)
    for i, h in enumerate(hit):
        run = 0
        for j, v in enumerate(h):
            run = run + 1 if v else 0
            if run >= k:
                out[i] = T[j]; break
    return out


def choose_threshold(T, S_val, y_val, target_far=0.02, k=1, grid=None):
    """Plus petit seuil dont le taux de fausses alertes sur les bénins de validation <= target_far.
    (Plus le seuil est bas, plus on détecte tôt : on prend le plus bas qui reste acceptable.)"""
    grid = np.linspace(0.05, 0.995, 190) if grid is None else grid
    ben = np.asarray(y_val) == 0
    for thr in grid:
        far = np.mean(~np.isnan(first_alarm(T, S_val[ben], thr, k)))
        if far <= target_far:
            return float(thr)
    return float(grid[-1])


def online_summary(T, S, y, lengths_run, thr, k=1):
    """Résumé de la détection en ligne sur un jeu : taux de détection, temps avant détection,
    fausses alertes."""
    t_alarm = first_alarm(T, S, thr, k)
    y = np.asarray(y).astype(int)
    rans, ben = y == 1, y == 0
    det = ~np.isnan(t_alarm[rans])
    tt = t_alarm[rans][det]
    frac = (t_alarm[rans] / lengths_run[rans])[det]
    return dict(
        n_rans=int(rans.sum()), n_benin=int(ben.sum()),
        taux_detection=float(det.mean()) if rans.any() else np.nan,
        temps_median_s=float(np.median(tt)) if len(tt) else np.nan,
        temps_moyen_s=float(np.mean(tt)) if len(tt) else np.nan,
        fraction_mediane=float(np.median(frac)) if len(frac) else np.nan,
        detectes_30s=float(np.mean(t_alarm[rans] <= 30)) if rans.any() else np.nan,
        fausses_alertes=float(np.mean(~np.isnan(t_alarm[ben]))) if ben.any() else np.nan,
    )
