"""Tâche 8 : références classiques (Random Forest, Gradient Boosting) avec Spark MLlib.

Ces modèles ne lisent pas une séquence : pour chaque exécution et chaque préfixe
(10/25/50/75/100 %), on résume les fenêtres observées par 4 statistiques par feature
(moyenne, écart-type, min, max) -> un vecteur de 24 x 4 = 96 valeurs.
C'est le point de comparaison « non séquentiel » des CNN / LSTM / GRU.
"""
import numpy as np
import pandas as pd
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler

from src.sequences.build import prefix_lengths

STATS = ["mean", "std", "min", "max"]


# ---------- 1. un vecteur par exécution et par préfixe ----------
def prefix_aggregates(X, lengths, frac, feats) -> pd.DataFrame:
    """Statistiques sur les ceil(frac * L) premières fenêtres de chaque exécution."""
    lp = prefix_lengths(lengths, frac)
    mask = np.arange(X.shape[1])[None, :, None] < lp[:, None, None]       # (N, T, 1)
    n = lp[:, None].astype(np.float64)
    s = np.where(mask, X, 0).sum(1)
    mean = s / n
    var = np.where(mask, (X - mean[:, None, :]) ** 2, 0).sum(1) / n
    mn = np.where(mask, X, np.inf).min(1)
    mx = np.where(mask, X, -np.inf).max(1)
    cols = {}
    for name, arr in zip(STATS, [mean, np.sqrt(var), mn, mx]):
        for j, f in enumerate(feats):
            cols[f"{f}__{name}"] = arr[:, j]
    return pd.DataFrame(cols)


def to_spark(spark, agg: pd.DataFrame, runs: pd.DataFrame):
    pdf = agg.assign(label=runs["label"].astype(float).values, split=runs["split"].values,
                     rel=runs["rel"].values, family=runs["family"].values)
    df = spark.createDataFrame(pdf)
    feat_cols = list(agg.columns)
    return VectorAssembler(inputCols=feat_cols, outputCol="features").transform(df), feat_cols


# ---------- 2. modèles ----------
def make_model(name, depth, seed=42):
    if name == "RF":
        return RandomForestClassifier(numTrees=200, maxDepth=depth, seed=seed,
                                      featureSubsetStrategy="sqrt")
    if name == "GBT":
        return GBTClassifier(maxIter=100, maxDepth=depth, stepSize=0.1, seed=seed)
    raise ValueError(name)


def predict(model, df) -> pd.DataFrame:
    """Probabilité « ransomware » pour chaque exécution."""
    from pyspark.ml.functions import vector_to_array
    out = (model.transform(df)
           .select("rel", "split", "family", "label",
                   vector_to_array("probability")[1].alias("p"))
           .toPandas())
    return out


# ---------- 3. métriques (numpy, sans sklearn) ----------
def _auc_roc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.Series(np.r_[pos, neg]).rank().values
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _auc_pr(y, p):
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())          # précision moyenne


def metrics(y, p, threshold=0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    yhat = (p >= threshold).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and not np.isnan(prec + rec) else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    return dict(n=len(y), accuracy=(tp + tn) / len(y), precision=prec, recall=rec, f1=f1,
                fpr=fpr, auc_roc=_auc_roc(y, p), auc_pr=_auc_pr(y, p), tp=tp, fp=fp, fn=fn, tn=tn)


# ---------- 4. une expérience complète ----------
def run_fraction(spark, X, lengths, runs, feats, frac, models=("RF", "GBT"),
                 depths={"RF": [5, 10], "GBT": [3, 5]}, eval_splits=None):
    """Pour une fraction : choisit la profondeur sur val (F1), réentraîne rien d'autre,
    évalue sur chaque jeu de test. Renvoie (résultats, modèles retenus)."""
    eval_splits = eval_splits or ["val", "test", "test_hw", "test_mix", "test_variants"]
    df, feat_cols = to_spark(spark, prefix_aggregates(X, lengths, frac, feats), runs)
    df = df.cache()
    train = df.where(df.split == "train")
    rows, best = [], {}
    for name in models:
        cands = []
        for d in depths[name]:
            m = make_model(name, d).fit(train)
            pv = predict(m, df.where(df.split == "val"))
            cands.append((metrics(pv.label, pv.p)["f1"], d, m))
        f1v, d, m = max(cands, key=lambda c: c[0])
        best[name] = (m, feat_cols)
        pred = predict(m, df.where(df.split.isin(eval_splits)))
        for sp, g in pred.groupby("split"):
            rows.append(dict(model=name, fraction=frac, depth=d, split=sp, **metrics(g.label, g.p)))
    df.unpersist()
    return pd.DataFrame(rows), best


def importances(model, feat_cols, top=15) -> pd.Series:
    return (pd.Series(model.featureImportances.toArray(), index=feat_cols)
            .sort_values(ascending=False).head(top))
