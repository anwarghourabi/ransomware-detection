# %% [markdown]
# # Tâche 8 : références classiques (Random Forest, Gradient Boosting, Spark MLlib)
# Entrée : data/processed/sequences (tâche 6). Pour chaque préfixe 10/25/50/75/100 %,
# chaque exécution est résumée par moyenne / écart-type / min / max de ses 24 features.

# %% Cellule 1 : chemins et imports
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from src.models import baselines as M
from src.sequences import build as B

SEQ = ROOT / "data" / "processed" / "sequences"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"

X, lengths, labels, runs = B.load(SEQ)
feats = json.loads((SEQ / "preprocessing.json").read_text())["features"]
print(X.shape, "|", runs.split.value_counts().to_dict())

# %% Cellule 2 : Spark
from src.utils.spark_session import get_spark

spark = get_spark()
spark.sparkContext.setLogLevel("ERROR")

# %% Cellule 3 : entraînement + évaluation pour chaque fraction (~10-20 min)
# Profondeur choisie sur val (F1) ; test, test_hw, test_mix, test_variants jamais vus.
results, models = [], {}
for frac in B.FRACTIONS:
    t = time.time()
    res, best = M.run_fraction(spark, X, lengths, runs, feats, frac)
    results.append(res); models[frac] = best
    print(f"{int(frac * 100):3d} % terminé en {time.time() - t:.0f} s")
results = pd.concat(results, ignore_index=True)
results.to_csv(REPORTS / "baseline_results.csv", index=False)

# %% Cellule 4 : tableau principal (jeu de test original) -> RQ1 / RQ2
cols = ["accuracy", "precision", "recall", "f1", "fpr", "auc_roc", "auc_pr"]
display(results[results.split == "test"]
        .pivot_table(index="fraction", columns="model", values=cols).round(3))

# %% Cellule 5 : généralisation (matériel jamais vu, mix, variantes)
display(results[results.split == "test_hw"].pivot_table(index="fraction", columns="model",
                                                        values=["f1", "recall", "fpr"]).round(3))
# mix et variants ne contiennent que des ransomwares : seul le rappel (taux de détection) a un sens
display(results[results.split.isin(["test_mix", "test_variants"])]
        .pivot_table(index="fraction", columns=["split", "model"], values="recall").round(3))

# %% Cellule 6 : figure : qualité selon la fraction observée
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for name, c in [("RF", "tab:green"), ("GBT", "tab:purple")]:
    d = results[(results.model == name) & (results.split == "test")].sort_values("fraction")
    x = d.fraction * 100
    ax[0].plot(x, d.f1, "o-", color=c, label=name)
    ax[1].plot(x, d.fpr, "o-", color=c, label=name)
    for sp, ls in [("test_mix", "--"), ("test_variants", ":")]:
        e = results[(results.model == name) & (results.split == sp)].sort_values("fraction")
        ax[2].plot(e.fraction * 100, e.recall, ls, marker="o", color=c, label=f"{name} {sp}")
ax[0].set(title="F1 (test)", xlabel="% de l'exécution observée", ylim=(0, 1.02))
ax[1].set(title="Taux de faux positifs (test)", xlabel="% de l'exécution observée")
ax[2].set(title="Rappel : mix et variantes", xlabel="% de l'exécution observée", ylim=(0, 1.02))
for a in ax: a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "fig_baselines_fraction.png", dpi=150); plt.show()

# %% Cellule 7 : quelles features comptent ? (Random Forest, préfixe 25 %)
m, feat_cols = models[0.25]["RF"]
imp = M.importances(m, feat_cols)
display(imp.round(3))
imp[::-1].plot.barh(figsize=(7, 5)); plt.title("Importance des features (RF, préfixe 25 %)")
plt.tight_layout(); plt.savefig(FIG / "fig_baselines_importance.png", dpi=150); plt.show()

# %%
