# %% [markdown]
# # Tâche 5 : caractéristiques par fenêtre temporelle + choix de la fenêtre
# Parquet (tâche 4, un événement par ligne) -> une ligne par (exécution, fenêtre de W secondes).
# Exécuter cellule par cellule.

# %% Cellule 1 : chemins et imports
import os
import sys
import time
from pathlib import Path
from IPython.display import display

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import matplotlib.pyplot as plt
import pandas as pd

from src.features.windows import PREFIX, build_features, feature_columns, first_ts_from_zip, offsets
from src.ingestion.ingest import folder_size_gb

ZIP = ROOT / "data" / "raw" / "archive.zip"
CSV_DIR = ROOT / "data" / "raw" / "ransmap"
PARQUET = ROOT / "data" / "processed" / "parquet"
FEATURES = ROOT / "data" / "processed" / "features"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
TRACES = list(PREFIX)            # ata_read, ata_write, mem_read, mem_write

runs = pd.read_csv(REPORTS / "runs_labels.csv")   # écrit en tâche 4
print(len(runs), "exécutions")

# %% Cellule 2 : Spark
from pyspark.sql import functions as F
from src.utils.spark_session import get_spark

spark = get_spark()
spark.sparkContext.setLogLevel("ERROR")
spark.conf.set("spark.sql.shuffle.partitions", "64")

# %% Cellule 3 : aligner les 4 traces de chaque exécution (une seule fois, ~2-5 min)
# En tâche 4, t part de 0 pour chaque fichier séparément. Si les écritures disque commencent
# 3 s après les accès mémoire, il faut les décaler de 3 s pour que la fenêtre n°w soit
# la même seconde pour les 4 traces. On lit le début de chaque CSV dans le zip.
OFFSETS = ROOT / "data" / "processed" / "trace_offsets.csv"
if OFFSETS.exists():
    offs = pd.read_csv(OFFSETS)
else:
    t = time.time()
    offs = offsets(first_ts_from_zip(ZIP, TRACES))
    offs.to_csv(OFFSETS, index=False)
    print(f"{len(offs)} fichiers lus en {time.time() - t:.0f} s")
display(offs.groupby("trace")["offset"].describe().round(2))

# %% Cellule 4 : contrôle : le début du fichier contient bien le premier instant ?
# On compare, sur quelques CSV encore présents (config de la tâche 3), le minimum des 1000
# premières lignes au minimum du fichier entier. Les deux colonnes doivent être égales.
check = []
for rel in runs[(runs.category == "original") & (runs.cpu == "i3-gen12") & (runs.ram == "ddr4-2133-16g")].groupby("sample").head(1).rel:
    for tr in ["ata_write", "mem_read"]:
        f = CSV_DIR / rel / f"{tr}.csv"
        if not f.exists() or f.stat().st_size == 0:
            continue
        d = pd.read_csv(f, header=None, usecols=[0, 1], names=["sec", "ns"])
        ts = d.sec + d.ns / 1e9
        check.append((rel.split("/")[-2], tr, ts.head(1000).min(), ts.min()))
check = pd.DataFrame(check, columns=["sample", "trace", "min_1000_lignes", "min_fichier"])
check["ok"] = (check.min_1000_lignes - check.min_fichier).abs() < 1e-3
display(check)

# %% Cellule 5 : choix de la taille de fenêtre (sur une configuration seulement)
SUB = dict(category="original", cpu="i3-gen12", ram="ddr4-2133-16g")
sub_runs = runs[(runs.category == SUB["category"]) & (runs.cpu == SUB["cpu"]) & (runs.ram == SUB["ram"])]
choice = []
for W in [0.1, 0.5, 1.0, 2.0, 5.0]:
    t = time.time()
    f = build_features(spark, PARQUET, sub_runs, offs, W, subset=SUB).toPandas()
    sec = time.time() - t
    per_run = f.groupby("rel").agg(n_win=("w", "max"), aw_actives=("aw_n", lambda s: (s > 0).sum()))
    per_run["n_win"] += 1
    choice.append(dict(W=W, temps_s=round(sec, 1),
                       fenetres_par_execution=int(per_run.n_win.median()),
                       part_fenetres_avec_ecriture_disque=round((per_run.aw_actives / per_run.n_win).median(), 2),
                       fenetres_a_10pct=int(per_run.n_win.median() * 0.1)))
choice = pd.DataFrame(choice)
choice.to_csv(REPORTS / "choix_fenetre.csv", index=False)
choice
# Lecture : il faut assez de fenêtres à 10 % pour qu'un LSTM / CNN ait une séquence (>= 10),
# mais pas des milliers de pas de temps presque vides. W = 1 s donne ~200 pas par exécution
# et ~20 à 10 % : bon compromis. W = 0,1 s -> ~2000 pas, majorité vides ; W = 5 s -> 4 pas à 10 %.

# %% Cellule 6 : features sur tout le dataset avec la fenêtre choisie (~20-40 min)
W = 1.0
OUT = FEATURES / f"w={W}"
t = time.time()
feats = build_features(spark, PARQUET, runs, offs, W)
(feats.repartition("category").write.mode("overwrite")
      .partitionBy("category").parquet(OUT.as_posix()))
print(f"Features écrites en {(time.time() - t) / 60:.1f} min -> {OUT} ({folder_size_gb(OUT):.2f} Go)")

# %% Cellule 7 : vérification
fe = spark.read.parquet(OUT.as_posix())
fe.printSchema()
print("lignes :", fe.count(), "| exécutions :", fe.select("rel").distinct().count(), "(attendu", len(runs), ")")
print(len(feature_columns(fe)), "features :", feature_columns(fe))
dur = fe.groupBy("rel").agg((F.max("w") + 1).alias("n_windows")).toPandas()
dur = runs[["rel", "category", "sample", "family", "label"]].merge(dur, on="rel", how="left")
dur.to_csv(REPORTS / "run_windows.csv", index=False)      # utile en tâche 6 pour les préfixes
display(dur.groupby("sample")["n_windows"].describe()[["min", "50%", "max"]])

# %% Cellule 8 : figure : évolution moyenne dans le temps, ransomware vs bénin (config de la tâche 3)
pdf = (fe.where((F.col("cpu") == SUB["cpu"]) & (F.col("ram") == SUB["ram"]) & (F.col("category") == "original"))
         .groupBy("label", "w").agg(F.avg("aw_n").alias("aw_n"), F.avg("aw_ent_hi").alias("aw_ent_hi"),
                                    F.avg("mw_n").alias("mw_n"), F.avg("ar_n").alias("ar_n"))
         .orderBy("w").toPandas())
fig, ax = plt.subplots(2, 2, figsize=(13, 6), sharex=True)
for a, col, title in zip(ax.flat, ["aw_n", "aw_ent_hi", "ar_n", "mw_n"],
                         ["écritures disque / s", "part des écritures à entropie élevée",
                          "lectures disque / s", "écritures mémoire / s"]):
    for lab, name, c in [(1, "ransomware", "tab:red"), (0, "bénin", "tab:blue")]:
        d = pdf[pdf.label == lab]
        a.plot(d.w, d[col], label=name, color=c)
    a.set_title(title)
ax[0, 0].legend(); ax[1, 0].set_xlabel("secondes"); ax[1, 1].set_xlabel("secondes")
plt.tight_layout(); plt.savefig(FIG / "fig_features_temps.png", dpi=150); plt.show()

#%%
display(dur[dur.n_windows < 100])
# %%
