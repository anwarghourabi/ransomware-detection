# %% [markdown]
# # Tâche 11 : RQ1, performance selon la fraction observée (10 / 25 / 50 / 75 / 100 %)
# Question : avec seulement le début d'une exécution, détecte-t-on déjà le ransomware ?
# Trois protocoles (cf. plan) :
# - A. « modèle 100 % » : les modèles des tâches 9-10 (entraînés sur la séquence complète),
#      appliqués tels quels aux préfixes. Aucun entraînement, quelques secondes.
# - B. « un modèle par fraction » : on entraîne un modèle sur les préfixes f et on le teste sur f.
# - C. « modèle unique multi-préfixes » : un seul modèle, entraîné sur des préfixes tirés au hasard
#      (10 à 100 %), testé sur chaque fraction.
# RF / GBT (tâche 8) sont déjà « un modèle par fraction » : ils servent de référence.

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
PARTS = REPORTS / "rq1_parts"; PARTS.mkdir(exist_ok=True)      # un CSV par entraînement (reprise possible)

X, lengths, labels, runs = B.load(SEQ)
X = X[:, :lengths.max()]
y = labels.astype(np.float32)
idx = {s: np.where(runs.split == s)[0] for s in runs.split.unique()}
EVAL = ["val", "test", "test_hw", "test_mix", "test_variants"]
FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]
MODEL_NAMES = ["CNN", "LSTM", "GRU", "MHSA-LSTM"]    # MHSA-LSTM-IG écarté (tâche 10 : pas de gain)
SEEDS = [42, 43, 44]
print(X.shape, "| longueur médiane :", int(np.median(lengths)), "fenêtres")
print({f: int(np.median(B.prefix_lengths(lengths, f))) for f in FRACS}, "<- fenêtres médianes par fraction")


def evaluate(m, name, protocol, seed, frac):
    lp = B.prefix_lengths(lengths, frac)
    return [dict(model=name, protocol=protocol, seed=seed, fraction=frac, split=sp,
                 **metrics(y[idx[sp]], D.predict_proba(m, X, lp, idx[sp]))) for sp in EVAL]


# %% Cellule 2 : protocole A, modèles 100 % (tâches 9-10) appliqués aux préfixes (~1 min)
rows = []
for name in MODEL_NAMES:
    for seed in SEEDS:
        f = MODELS / f"{name}_100_seed{seed}.pt"
        if not f.exists():
            print("absent :", f.name); continue
        m = D.make_model(name, X.shape[2]); m.load_state_dict(torch.load(f))
        for frac in FRACS:
            rows += evaluate(m, name, "A_modele_100", seed, frac)
resA = pd.DataFrame(rows)
resA.to_csv(PARTS / "A_modele_100.csv", index=False)
print(resA[resA.split == "test"].pivot_table(index="fraction", columns="model", values="f1").round(3))

# %% Cellule 3 : test rapide du temps (LSTM, 10 %, 2 époques)
lp = B.prefix_lengths(lengths, 0.1)
m = D.train("LSTM", X, lp, y, idx["train"], idx["val"], epochs=2)

# %% Cellule 4 : protocole B, un modèle par fraction (le plus long ; reprend là où ça s'est arrêté)
# 100 % est déjà fait (tâches 9-10) : on ne refait que 10 / 25 / 50 / 75 %.
for frac in [0.1, 0.25, 0.5, 0.75]:
    lp = B.prefix_lengths(lengths, frac)
    for name in MODEL_NAMES:
        for seed in SEEDS:
            out = PARTS / f"B_{name}_{int(frac * 100)}_seed{seed}.csv"
            if out.exists():
                continue
            t = time.time()
            m = D.train(name, X, lp, y, idx["train"], idx["val"], seed=seed, verbose=False)
            torch.save(m.state_dict(), MODELS / f"{name}_{int(frac * 100)}_seed{seed}.pt")
            pd.DataFrame(evaluate(m, name, "B_par_fraction", seed, frac)).to_csv(out, index=False)
            print(f"{int(frac * 100):3d} %  {name:9s} graine {seed} : {time.time() - t:.0f} s")

# %% Cellule 5 : protocole C, un modèle unique multi-préfixes (reprend là où ça s'est arrêté)
for name in MODEL_NAMES:
    for seed in SEEDS:
        out = PARTS / f"C_{name}_seed{seed}.csv"
        if out.exists():
            continue
        t = time.time()
        m = D.train(name, X, lengths, y, idx["train"], idx["val"], seed=seed, verbose=False,
                    prefix_fracs=FRACS)
        torch.save(m.state_dict(), MODELS / f"{name}_multi_seed{seed}.pt")
        rows = [r for frac in FRACS for r in evaluate(m, name, "C_multi_prefixes", seed, frac)]
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"multi  {name:9s} graine {seed} : {time.time() - t:.0f} s")

# %% Cellule 6 : rassembler tout -> reports/rq1_results.csv
parts = [pd.read_csv(f) for f in sorted(PARTS.glob("*.csv"))]
# B à 100 % = résultats des tâches 9 et 10
for f in ["deep_results_100.csv", "attention_results_100.csv"]:
    if (REPORTS / f).exists():
        d = pd.read_csv(REPORTS / f); parts.append(d[d.model.isin(MODEL_NAMES)].assign(protocol="B_par_fraction"))
base = pd.read_csv(REPORTS / "baseline_results.csv").assign(protocol="B_par_fraction")
parts.append(base)
res = pd.concat(parts, ignore_index=True)
res.to_csv(REPORTS / "rq1_results.csv", index=False)
print(res.groupby(["protocol", "model"]).size())

# %% Cellule 7 : TABLEAU RQ1 (protocole B, moyenne sur 3 graines)
ORDER = ["RF", "GBT", "CNN", "LSTM", "GRU", "MHSA-LSTM"]
B_ = res[res.protocol == "B_par_fraction"]
tab = {}
for sp, col in [("test", "f1"), ("test", "fpr"), ("test_hw", "f1"), ("test_mix", "recall"), ("test_variants", "recall")]:
    tab[f"{col} {sp}"] = (B_[B_.split == sp].groupby(["fraction", "model"])[col].mean()
                          .unstack()[[m for m in ORDER if m in B_.model.unique()]])
rq1 = pd.concat(tab, axis=1).round(3)
rq1.to_csv(REPORTS / "tableau_rq1.csv")
for k in tab:
    print("\n" + k); display(tab[k].round(3))

# %% Cellule 8 : comparaison des 3 protocoles (rappel mix / variantes, F1 test)
P = res[res.model.isin(MODEL_NAMES)]
cmp = (P[P.split.isin(["test", "test_mix", "test_variants"])]
       .assign(val=lambda d: np.where(d.split == "test", d.f1, d.recall))
       .pivot_table(index=["model", "fraction"], columns=["split", "protocol"], values="val").round(3))
display(cmp)

# %% Cellule 9 : figure RQ1 (protocole B)
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
pct = lambda s: (s.index * 100)
for m in [m for m in ORDER if m in B_.model.unique()]:
    st = "--" if m in ("RF", "GBT") else "-"
    ax[0].plot(pct(tab["f1 test"][m]), tab["f1 test"][m], st, marker="o", label=m)
    ax[1].plot(pct(tab["fpr test"][m]), tab["fpr test"][m], st, marker="o", label=m)
    ax[2].plot(pct(tab["recall test_mix"][m]), tab["recall test_mix"][m], st, marker="o", label=m)
    ax[3].plot(pct(tab["recall test_variants"][m]), tab["recall test_variants"][m], st, marker="o", label=m)
for a, t in zip(ax, ["F1 (test)", "Faux positifs (test)", "Rappel test_mix", "Rappel test_variants"]):
    a.set(title=t, xlabel="% de l'exécution observée"); a.grid(alpha=.3)
ax[0].set_ylim(0.5, 1.01); ax[2].set_ylim(0, 1.01); ax[3].set_ylim(0, 1.01); ax[0].legend()
plt.tight_layout(); plt.savefig(FIG / "fig_rq1_fraction.png", dpi=150); plt.show()

# %% Cellule 10 : figure protocoles (LSTM) : par fraction vs 100 % vs multi-préfixes
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
L = P[P.model == "LSTM"]
for a, (sp, col) in zip(ax, [("test", "f1"), ("test_mix", "recall"), ("test_variants", "recall")]):
    for proto, d in L[L.split == sp].groupby("protocol"):
        s = d.groupby("fraction")[col].mean()
        a.plot(s.index * 100, s, marker="o", label=proto)
    a.set(title=f"LSTM : {col} {sp}", xlabel="% observé", ylim=(0, 1.01)); a.grid(alpha=.3)
ax[0].legend()
plt.tight_layout(); plt.savefig(FIG / "fig_rq1_protocoles.png", dpi=150); plt.show()

# %%
