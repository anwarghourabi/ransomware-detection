# %% [markdown]
# # Tâche 6 : séquences, préfixes 10/25/50/75/100 %, découpages train/test figés
# Entrée : data/processed/features/w=1.0 (tâche 5). Sortie : data/processed/sequences/

# %% Cellule 1 : chemins et imports
import os
import sys
from pathlib import Path
from IPython.display import display
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.sequences import build as B

W = 1.0
FEATURES = ROOT / "data" / "processed" / "features" / f"w={W}"
OUT = ROOT / "data" / "processed" / "sequences"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"

# %% Cellule 2 : charger la table de features (pandas, sans Spark)
fe = pd.read_parquet(FEATURES)
fe["category"] = fe["category"].astype(str)
feats_all = [c for c in fe.columns if c not in B.META and c != "w"]
print(fe.shape, "|", fe.rel.nunique(), "exécutions")

# %% Cellule 3 : écarter les exécutions incomplètes et les features constantes
fe, short = B.keep_runs(fe)
print("écartées (<", B.MIN_WINDOWS, "fenêtres) :", dict(short))
feats = B.useful_features(fe, feats_all)
print(len(feats), "features gardées ; retirées :", sorted(set(feats_all) - set(feats)))

# %% Cellule 4 : séquences (une matrice fenêtres x features par exécution)
X, lengths, runs = B.to_sequences(fe, feats)
X = B.log_transform(X, feats)
print("X :", X.shape, "(exécutions, fenêtres max, features) |", round(X.nbytes / 1e6), "Mo")
print("longueurs : min", lengths.min(), "médiane", int(np.median(lengths)), "max", lengths.max())

# %% Cellule 5 : découpages figés (par exécution, jamais par fenêtre)
runs["split"] = B.make_splits(runs).values
display(pd.crosstab(runs.split, runs.label, margins=True))
orig = runs[runs.category == "original"]
display(pd.crosstab(orig["sample"], orig["split"]))

# %% Cellule 6 : normalisation apprise sur le TRAIN seulement, puis sauvegarde
mean, std = B.fit_scaler(X, lengths, (runs.split == "train").to_numpy())
X = B.apply_scaler(X, lengths, mean, std)
B.save(OUT, X, lengths, runs, feats, mean, std)
runs.to_csv(REPORTS / "splits.csv", index=False)   # dans Git : Ghada a exactement les mêmes découpages
print("sauvegardé :", sorted(p.name for p in OUT.iterdir()))

# %% Cellule 7 : préfixes 10/25/50/75/100 % -> combien de fenêtres le modèle voit
tab = pd.DataFrame({f"{int(f * 100)} %": B.prefix_lengths(lengths, f) for f in B.FRACTIONS})
display(tab.describe().loc[["min", "50%", "max"]].astype(int))
Xp, lp = B.prefix(X, lengths, 0.25)          # exemple d'utilisation (pour les modèles)
print("préfixe 25 % :", Xp.shape, "| fenêtres vues (médiane) :", int(np.median(lp)))

# %% Cellule 8 : RQ3 : plis leave-one-family-out (Conti + ses variantes ensemble)
folds = B.lofo_folds(runs)
display(pd.DataFrame({f: dict(train=len(tr), test=len(te),
                               test_ransomware=int(runs.label.iloc[te].sum()))
                      for f, (tr, te) in folds.items()}).T)

# %% Cellule 9 : contrôle visuel : une séquence ransomware vs bénigne (préfixe 25 % en gris)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.5), sharey=True)
j = feats.index("mw_n")
for a, s in zip(ax, ["Conti", "AESCrypt"]):
    i = runs.index[(runs["sample"] == s) & (runs.split == "train")][0]
    a.plot(X[i, :lengths[i], j]); a.set_title(f"{s} : écritures mémoire (normalisées)")
    a.axvspan(0, B.prefix_lengths(lengths[i:i + 1], 0.25)[0], color="grey", alpha=0.2)
    a.set_xlabel("fenêtre (s)")
plt.tight_layout(); plt.savefig(FIG / "fig_sequences_prefixe.png", dpi=150); plt.show()

# %% Cellule 10 : comment Ghada (ou la tâche 8/9) recharge les données
X, lengths, labels, runs = B.load(OUT)
train = (runs.split == "train").to_numpy()
X25, l25 = B.prefix(X, lengths, 0.25)
print(X25[train].shape, labels[train].mean().round(2))
