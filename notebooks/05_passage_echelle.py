# %% [markdown]
# # Tâche 7 : passage à l'échelle de Spark
# Temps de la vraie étape « features par fenêtre » (tâche 5) selon :
# (A) le nombre de cœurs, (B) le nombre de partitions, (C) la taille des données.
# Durée totale : ~30-60 min selon le PC. Garde le PC branché et ferme les autres applis.

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
import pandas as pd

from src.scaling.bench import bench

PARQUET = ROOT / "data" / "processed" / "parquet"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
offs = pd.read_csv(ROOT / "data" / "processed" / "trace_offsets.csv")   # tâche 5

N = os.cpu_count()
CORES = sorted({1, 2, 4, N // 2, N} - {0})
print("cœurs logiques :", N, "-> on teste", CORES)

# configurations (category, cpu, ram) utilisées comme charge de travail
CONFIGS = [("original", "i3-gen12", r) for r in
           ["ddr4-2133-16g", "ddr4-2133-8g", "ddr4-2666-16g", "ddr4-2666-8g"]]

# %% Cellule 2 : (A) temps selon le nombre de cœurs (2 configurations, partitions = 2 x cœurs max)
res_a = bench(PARQUET, offs, CONFIGS[:2], cores_list=CORES, shuffle_list=[2 * N])
res_a.to_csv(REPORTS / "scaling_coeurs.csv", index=False)

# %% Cellule 3 : (B) temps selon le nombre de partitions (tous les cœurs)
res_b = bench(PARQUET, offs, CONFIGS[:2], cores_list=[N], shuffle_list=[1, 4, 16, 64, 200, 800])
res_b.to_csv(REPORTS / "scaling_partitions.csv", index=False)

# %% Cellule 4 : (C) temps selon la taille des données (1, 2, 4 configurations, tous les cœurs)
res_c = pd.concat([bench(PARQUET, offs, CONFIGS[:k], cores_list=[N], shuffle_list=[2 * N])
                   for k in [1, 2, 4]])
res_c.to_csv(REPORTS / "scaling_taille.csv", index=False)

# %% Cellule 5 : tableau et figure
a = res_a.groupby("cores")["seconds"].mean()
speedup = a.iloc[0] / a
display(pd.DataFrame({"temps_s": a.round(1), "accélération": speedup.round(2),
                      "efficacité": (speedup / a.index).round(2)}))
b = res_b.groupby("shuffle_partitions")["seconds"].mean()
c = res_c.groupby("n_configs")["seconds"].mean()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(a.index, speedup, "o-", label="mesurée")
ax[0].plot(a.index, a.index, "--", color="grey", label="idéale")
ax[0].set(xlabel="cœurs", ylabel="accélération (temps 1 cœur / temps n cœurs)", title="(A) cœurs")
ax[0].legend()
ax[1].plot(b.index, b.values, "o-"); ax[1].set_xscale("log")
ax[1].set(xlabel="partitions de shuffle", ylabel="temps (s)", title=f"(B) partitions ({N} cœurs)")
ax[2].plot(c.index * 120, c.values, "o-")
ax[2].set(xlabel="exécutions traitées", ylabel="temps (s)", title=f"(C) taille des données ({N} cœurs)")
plt.tight_layout(); plt.savefig(FIG / "fig_passage_echelle.png", dpi=150); plt.show()
