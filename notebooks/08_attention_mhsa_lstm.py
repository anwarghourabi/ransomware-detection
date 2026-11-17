# %% [markdown]
# # Tâche 10 : LSTM + attention multi-têtes (MHSA-LSTM), avec et sans gain d'information
# Entrée : data/processed/sequences (tâche 6). Même découpage et mêmes métriques que les tâches 8 et 9.
# - MHSA-LSTM    : LSTM puis attention multi-têtes causale (ne regarde jamais le futur)
# - MHSA-LSTM-IG : pareil, mais chaque feature est pondérée par son gain d'information (calculé sur le train)

# %% Cellule 1 : chemins et imports
import json
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
feats = json.loads((SEQ / "preprocessing.json").read_text())["features"]
idx = {s: np.where(runs.split == s)[0] for s in runs.split.unique()}
print(X.shape, {k: len(v) for k, v in idx.items()})

# %% Cellule 2 : gain d'information de chaque feature (train seulement)
ig = D.info_gain(X, lengths, y, idx["train"])
W = D.ig_weights(ig)
ig_tab = pd.DataFrame({"feature": feats, "info_gain": ig.round(3), "poids": W.round(2)})
ig_tab = ig_tab.sort_values("info_gain", ascending=False)
ig_tab.to_csv(REPORTS / "info_gain.csv", index=False)
display(ig_tab)
ax = ig_tab.plot.barh(x="feature", y="info_gain", figsize=(7, 7), legend=False)
ax.invert_yaxis(); ax.set(title="Gain d'information par feature (train)", xlabel="bits")
plt.tight_layout(); plt.savefig(FIG / "fig_info_gain.png", dpi=150); plt.show()

# %% Cellule 3 : test rapide (2 époques) pour mesurer le temps par époque
m = D.train("MHSA-LSTM", X, lengths, y, idx["train"], idx["val"], epochs=2)
print(metrics(y[idx["test"]], D.predict_proba(m, X, lengths, idx["test"])))

# %% Cellule 4 : entraînement des 2 variantes × 3 graines (temps ≈ 2 × 3 × époques × temps de la cellule 3)
SEEDS = [42, 43, 44]
EVAL = ["val", "test", "test_hw", "test_mix", "test_variants"]
VARIANTS = {"MHSA-LSTM": {}, "MHSA-LSTM-IG": {"feat_weights": W}}
rows = []
for name, kw in VARIANTS.items():
    for seed in SEEDS:
        t = time.time()
        m = D.train(name, X, lengths, y, idx["train"], idx["val"], seed=seed, verbose=False, **kw)
        torch.save(m.state_dict(), MODELS / f"{name}_100_seed{seed}.pt")
        for sp in EVAL:
            p = D.predict_proba(m, X, lengths, idx[sp])
            rows.append(dict(model=name, seed=seed, fraction=1.0, split=sp, **metrics(y[idx[sp]], p)))
        print(f"{name} graine {seed} : {time.time() - t:.0f} s")
res = pd.DataFrame(rows)
res.to_csv(REPORTS / "attention_results_100.csv", index=False)

# %% Cellule 5 : comparaison avec RF / GBT (tâche 8) et CNN / LSTM / GRU (tâche 9), moyenne sur 3 graines
cols = ["f1", "fpr", "recall", "precision", "auc_roc"]
parts = [res, pd.read_csv(REPORTS / "deep_results_100.csv")]
base = pd.read_csv(REPORTS / "baseline_results.csv")
parts.append(base[base.fraction == 1.0])
allres = pd.concat(parts, ignore_index=True)
comp = allres.groupby(["split", "model"])[cols].mean().round(3)
display(comp)
# test_mix / test_variants : seulement des ransomwares -> regarder le rappel
print("Écart-type du rappel sur les 3 graines :")
display(res.groupby(["split", "model"]).recall.std().round(3).unstack())

# %% Cellule 6 : figure comparaison (F1 test / test_hw, rappel mix / variantes)
order = ["RF", "GBT", "CNN", "LSTM", "GRU", "MHSA-LSTM", "MHSA-LSTM-IG"]
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
comp["f1"].unstack(0).reindex(order)[["test", "test_hw"]].plot.bar(ax=ax[0], rot=20)
ax[0].set(title="F1 (100 % de la séquence)", ylim=(0.5, 1.01))
comp["recall"].unstack(0).reindex(order)[["test_mix", "test_variants"]].plot.bar(ax=ax[1], rot=20)
ax[1].set(title="Rappel : mix et variantes (100 %)", ylim=(0, 1.01))
for a in ax: a.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "fig_attention_100.png", dpi=150); plt.show()

# %% Cellule 7 : où regarde l'attention ? (modèle MHSA-LSTM-IG, graine 42, jeu de test)
m = D.make_model("MHSA-LSTM-IG", X.shape[2], feat_weights=W)
m.load_state_dict(torch.load(MODELS / "MHSA-LSTM-IG_100_seed42.pt"))
prof = D.attention_profile(m, X, lengths, idx["test"])
yt = y[idx["test"]]
pos = (np.arange(prof.shape[1]) + 0.5) / prof.shape[1] * 100
plt.figure(figsize=(8, 3.5))
plt.plot(pos, prof[yt == 1].mean(0), marker="o", label="ransomware")
plt.plot(pos, prof[yt == 0].mean(0), marker="o", label="bénin")
plt.xlabel("position dans la séquence (%)"); plt.ylabel("poids d'attention moyen")
plt.title("Attention de la dernière fenêtre selon la position"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "fig_attention_position.png", dpi=150); plt.show()

# %%
