# %% [markdown]
# # Tâche 9 : CNN 1D, LSTM, GRU sur la séquence complète (100 %)
# Entrée : data/processed/sequences (tâche 6). Même découpage et mêmes métriques que la tâche 8.

# %% Cellule 1 : chemins et imports
import sys
import time
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display

from src.models import deep as D
from src.models.baselines import metrics
from src.sequences import build as B

SEQ = ROOT / "data" / "processed" / "sequences"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)          # *.pt : pas dans Git

X, lengths, labels, runs = B.load(SEQ)
X = X[:, :lengths.max()]
y = labels.astype(np.float32)
idx = {s: np.where(runs.split == s)[0] for s in runs.split.unique()}
print(X.shape, {k: len(v) for k, v in idx.items()}, "| threads CPU :", torch.get_num_threads())

# %% Cellule 2 : test rapide (1 modèle, 2 époques) pour vérifier que tout marche
m = D.train("GRU", X, lengths, y, idx["train"], idx["val"], epochs=2)
p = D.predict_proba(m, X, lengths, idx["test"])
print(metrics(y[idx["test"]], p))

# %% Cellule 3 : entraînement des 3 modèles × 3 graines (~10-30 min sur CPU)
SEEDS = [42, 43, 44]
EVAL = ["val", "test", "test_hw", "test_mix", "test_variants"]
rows = []
for name in ["CNN", "LSTM", "GRU"]:
    for seed in SEEDS:
        t = time.time()
        m = D.train(name, X, lengths, y, idx["train"], idx["val"], seed=seed, verbose=False)
        torch.save(m.state_dict(), MODELS / f"{name}_100_seed{seed}.pt")
        for sp in EVAL:
            p = D.predict_proba(m, X, lengths, idx[sp])
            rows.append(dict(model=name, seed=seed, fraction=1.0, split=sp, **metrics(y[idx[sp]], p)))
        print(f"{name} graine {seed} : {time.time() - t:.0f} s")
res = pd.DataFrame(rows)
res.to_csv(REPORTS / "deep_results_100.csv", index=False)

# %% Cellule 4 : moyenne sur les 3 graines, comparée à Random Forest (tâche 8)
cols = ["f1", "fpr", "recall", "precision", "auc_roc"]
mean = res.groupby(["split", "model"])[cols].mean().round(3)
base = pd.read_csv(REPORTS / "baseline_results.csv")
base = base[(base.fraction == 1.0)].groupby(["split", "model"])[cols].mean().round(3)
comp = pd.concat([mean, base]).sort_index()
display(comp)
# test_mix / test_variants : seulement des ransomwares -> regarder le rappel

# %% Cellule 5 : figure : rappel sur les jeux difficiles, F1 sur test et test_hw
order = ["RF", "GBT", "CNN", "LSTM", "GRU"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
f1 = comp["f1"].unstack(0).reindex(order)[["test", "test_hw"]]
f1.plot.bar(ax=ax[0], rot=0); ax[0].set(title="F1 (100 % de la séquence)", ylim=(0.5, 1.01))
rec = comp["recall"].unstack(0).reindex(order)[["test_mix", "test_variants"]]
rec.plot.bar(ax=ax[1], rot=0); ax[1].set(title="Rappel : mix et variantes (100 %)", ylim=(0, 1.01))
for a in ax: a.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "fig_deep_100.png", dpi=150); plt.show()

# %%
print(res.groupby(["split","model"]).recall.std().round(3))

# %%
