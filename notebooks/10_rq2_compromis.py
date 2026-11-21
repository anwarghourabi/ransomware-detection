# %% [markdown]
# # Tâche 12 : RQ2, compromis précocité / qualité
# 1. Précision, rappel, F1, FPR selon la fraction observée (résultats de la tâche 11)
# 2. Courbes ROC et PR par fraction (LSTM)
# 3. Détection en ligne : toutes les 5 s, le modèle donne un score sur ce qu'il a vu depuis le début ;
#    alerte au premier score >= seuil (seuil choisi sur la validation : <= 2 % de fausses alertes).
#    -> temps avant détection, taux de détection, fausses alertes.

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

from src.evaluation import online as O
from src.models import deep as D
from src.sequences import build as B

SEQ = ROOT / "data" / "processed" / "sequences"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
MODELS = ROOT / "models"

X, lengths, labels, runs = B.load(SEQ)
X = X[:, :lengths.max()]
y = labels.astype(int)
idx = {s: np.where(runs.split == s)[0] for s in runs.split.unique()}
rq1 = pd.read_csv(REPORTS / "rq1_results.csv")
ORDER = ["RF", "GBT", "CNN", "LSTM", "GRU", "MHSA-LSTM"]
DEEP = ["CNN", "LSTM", "GRU", "MHSA-LSTM"]
print(X.shape, "| rq1 :", rq1.shape)

# %% Cellule 2 : vérification du protocole A (tâche 11) : taux de faux positifs sur le test
d = rq1[rq1.model.isin(DEEP) & (rq1.split == "test")]
display(d.pivot_table(index="fraction", columns=["protocol", "model"], values="fpr").round(3))

# %% Cellule 3 : précision, rappel, F1, FPR selon la fraction (protocole B, moyenne 3 graines)
Bres = rq1[rq1.protocol == "B_par_fraction"]
fig, ax = plt.subplots(2, 4, figsize=(19, 7.5), sharex=True)
for r, sp in enumerate(["test", "test_hw"]):
    for c, col in enumerate(["precision", "recall", "f1", "fpr"]):
        t = Bres[Bres.split == sp].groupby(["fraction", "model"])[col].mean().unstack()
        for m in [m for m in ORDER if m in t]:
            ax[r, c].plot(t.index * 100, t[m], "--" if m in ("RF", "GBT") else "-", marker="o", label=m)
        ax[r, c].set(title=f"{col} ({sp})"); ax[r, c].grid(alpha=.3)
        if col != "fpr":
            ax[r, c].set_ylim(0.8, 1.005)
for a in ax[1]: a.set_xlabel("% de l'exécution observée")
ax[0, 0].legend()
plt.tight_layout(); plt.savefig(FIG / "fig_rq2_metriques.png", dpi=150); plt.show()

# %% Cellule 4 : courbes ROC et PR par fraction (LSTM, protocole B, graine 42)
# « facile » = test (programmes connus) ; « difficile » = bénins du test + ransomwares mix et variantes
hard = np.r_[idx["test"][y[idx["test"]] == 0], idx["test_mix"], idx["test_variants"]]
fig, ax = plt.subplots(2, 2, figsize=(11, 9))
aucs = []
for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
    m = D.make_model("LSTM", X.shape[2])
    m.load_state_dict(torch.load(MODELS / f"LSTM_{int(frac * 100)}_seed42.pt"))
    lp = B.prefix_lengths(lengths, frac)
    for r, (nom, ii) in enumerate([("test", idx["test"]), ("difficile", hard)]):
        p = D.predict_proba(m, X, lp, ii)
        fpr, tpr = O.roc_curve(y[ii], p); rec, prec = O.pr_curve(y[ii], p)
        a_roc, a_pr = O.auc(fpr, tpr), O.auc(rec, prec)
        aucs.append(dict(fraction=frac, jeu=nom, auc_roc=a_roc, auc_pr=a_pr))
        ax[r, 0].plot(fpr, tpr, label=f"{int(frac * 100)} % (AUC {a_roc:.3f})")
        ax[r, 1].plot(rec, prec, label=f"{int(frac * 100)} % (AUC {a_pr:.3f})")
for r, nom in enumerate(["test", "difficile : bénins test + mix + variantes"]):
    ax[r, 0].set(title=f"ROC, {nom}", xlabel="taux de faux positifs", ylabel="rappel")
    ax[r, 1].set(title=f"Précision-rappel, {nom}", xlabel="rappel", ylabel="précision")
    for a in ax[r]: a.legend(fontsize=8); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "fig_rq2_roc_pr.png", dpi=150); plt.show()
display(pd.DataFrame(aucs).pivot(index="fraction", columns="jeu").round(3))

# %% Cellule 5 : détection en ligne : scores toutes les 5 s (~5-15 min)
# Modèles uniques capables de lire n'importe quelle longueur : protocole C (multi-préfixes) et A (modèle 100 %).
STEP = 5
ONLINE = {"LSTM (C)": ("LSTM", "multi"), "GRU (C)": ("GRU", "multi"), "CNN (C)": ("CNN", "multi"),
          "MHSA-LSTM (C)": ("MHSA-LSTM", "multi"), "LSTM (A)": ("LSTM", "100")}
SPLITS = ["val", "test", "test_hw", "test_mix", "test_variants"]
scores = {}
for label, (name, tag) in ONLINE.items():
    t0 = time.time()
    m = D.make_model(name, X.shape[2]); m.load_state_dict(torch.load(MODELS / f"{name}_{tag}_seed42.pt"))
    for sp in SPLITS:
        scores[label, sp] = O.online_scores(m, X, lengths, idx[sp], step=STEP, t_max=lengths.max())
    print(f"{label} : {time.time() - t0:.0f} s")

# %% Cellule 6 : seuil choisi sur la validation, puis temps avant détection
rows, THR = [], {}
for label in ONLINE:
    for k in [1, 2]:                                        # k = nombre de dépassements consécutifs exigés
        Tt, Sv = scores[label, "val"]
        thr = O.choose_threshold(Tt, Sv, y[idx["val"]], target_far=0.02, k=k)
        THR[label, k] = thr
        for sp in SPLITS[1:]:
            Tt, Ss = scores[label, sp]
            r = O.online_summary(Tt, Ss, y[idx[sp]], lengths[idx[sp]], thr, k)
            rows.append(dict(modele=label, k=k, seuil=round(thr, 3), split=sp, **r))
online = pd.DataFrame(rows)
online.to_csv(REPORTS / "rq2_online.csv", index=False)
cols = ["taux_detection", "temps_median_s", "fraction_mediane", "detectes_30s", "fausses_alertes"]
display(online[online.k == 1].set_index(["split", "modele"])[cols].round(3))
print("Avec k = 2 (deux dépassements de suite) :")
display(online[online.k == 2].set_index(["split", "modele"])[cols].round(3))

# %% Cellule 7 : figure : % de ransomwares détectés en fonction du temps (k = 1)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2), sharey=True)
grid_t = np.arange(0, lengths.max() + STEP, STEP)
for a, sp in zip(ax, ["test", "test_mix", "test_variants"]):
    for label in ONLINE:
        Tt, Ss = scores[label, sp]
        ta = O.first_alarm(Tt, Ss[y[idx[sp]] == 1], THR[label, 1])
        far = online[(online.modele == label) & (online.k == 1) & (online.split == "test")].fausses_alertes.iloc[0]
        a.step(grid_t, [(ta <= t).mean() for t in grid_t], where="post", label=f"{label} (FA test {far:.1%})")
    a.set(title=f"Ransomwares détectés : {sp}", xlabel="temps depuis le début (s)", xlim=(0, int(lengths.max()))); a.grid(alpha=.3)
ax[0].set_ylabel("part détectée (cumulée)"); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "fig_rq2_detection_temps.png", dpi=150); plt.show()

# %% Cellule 8 : compromis seuil : fausses alertes vs temps avant détection (LSTM (C), test)
Tt, Sb = scores["LSTM (C)", "test"]
yt = y[idx["test"]]; Lt = lengths[idx["test"]]
pts = []
for thr in np.linspace(0.1, 0.99, 40):
    r = O.online_summary(Tt, Sb, yt, Lt, thr)
    pts.append(dict(seuil=thr, **r))
pts = pd.DataFrame(pts)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(pts.fausses_alertes * 100, pts.temps_median_s, marker="o")
ax[0].set(xlabel="fausses alertes sur les bénins (%)", ylabel="temps médian avant détection (s)",
          title="LSTM (C), test : plus on veut détecter tôt, plus on a de fausses alertes")
ax[1].plot(pts.seuil, pts.taux_detection, label="taux de détection")
ax[1].plot(pts.seuil, pts.fausses_alertes, label="fausses alertes")
ax[1].axvline(THR["LSTM (C)", 1], ls="--", c="gray", label="seuil choisi (val)")
ax[1].set(xlabel="seuil", title="Effet du seuil"); ax[1].legend()
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "fig_rq2_compromis.png", dpi=150); plt.show()

# %%
