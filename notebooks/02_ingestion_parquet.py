# %% [markdown]
# # Tâche 4 : ingestion PySpark -> Parquet étiqueté
# CSV de RanSMAP -> data/processed/parquet/<trace>/category=../cpu=../ram=../family=../
# Exécuter cellule par cellule (« Run Cell »).

# %% Cellule 1 : chemins et imports
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pandas as pd

from src.ingestion.ingest import add_labels, chunks, folder_size_gb, ingest_chunk
from src.ingestion.ransmap_zip import USEFUL, extract_subset, inventory_from_zip

ZIP = ROOT / "data" / "raw" / "archive.zip"
CSV_DIR = ROOT / "data" / "raw" / "ransmap"            # CSV extraits (temporaires)
PARQUET = ROOT / "data" / "processed" / "parquet"      # sortie de la tâche 4
REPORTS = ROOT / "reports"
PARQUET.mkdir(parents=True, exist_ok=True)
print("Archive :", ZIP.exists(), "| Parquet ->", PARQUET)

# %% Cellule 2 : inventaire + étiquettes (table des exécutions)
inv = add_labels(inventory_from_zip(ZIP))
inv.to_csv(REPORTS / "runs_labels.csv", index=False)
print(len(inv), "exécutions")
display(inv.groupby(["category", "label"]).size().unstack(fill_value=0))
plan = chunks(inv, USEFUL)          # un morceau = (category, cpu, ram)
display(plan)
print("Total CSV utile :", round(plan.csv_go.sum(), 1), "Go")

# %% Cellule 3 : Spark
from pyspark.sql import functions as F
from src.utils.spark_session import get_spark

spark = get_spark()
spark.sparkContext.setLogLevel("ERROR")
print(spark.version, spark.sparkContext.master)

# %% Cellule 4 : premier morceau = la configuration déjà extraite en tâche 3
CHUNK = dict(category="original", cpu="i3-gen12", ram="ddr4-2133-16g")
runs = inv[(inv.category == CHUNK["category"]) & (inv.cpu == CHUNK["cpu"]) & (inv.ram == CHUNK["ram"])]
extract_subset(ZIP, CSV_DIR, runs)   # ne fait rien si déjà extrait
print(len(runs), "exécutions ;", round(folder_size_gb(CSV_DIR / CHUNK["category"] / CHUNK["cpu"] / CHUNK["ram"]), 2), "Go de CSV")

t = time.time()
ingest_chunk(spark, CSV_DIR, runs, PARQUET, USEFUL)
print(f"Ingestion terminée en {(time.time() - t) / 60:.1f} min")
with open(PARQUET / "_chunks_done.txt", "a") as f:     # pour que la cellule 8 le saute
    f.write("{category}/{cpu}/{ram}\n".format(**CHUNK))

# %% Cellule 5 : vérification
ata = spark.read.parquet((PARQUET / "ata_write").as_posix())
ata.printSchema()
ata.show(5, truncate=False)
print("exécutions :", ata.select("rel").distinct().count(), "(attendu", len(runs), ")")
ata.groupBy("category", "family", "label").agg(
    F.countDistinct("rel").alias("executions"), F.count("*").alias("evenements"),
    F.round(F.max("t"), 1).alias("t_max_s")).orderBy("label", "family").show(50)
ata.where(F.col("t") < 0).count()   # doit être 0

# %% Cellule 6 : CSV vs Parquet (taille et vitesse) -> pour le rapport
csv_go = folder_size_gb(CSV_DIR / CHUNK["category"] / CHUNK["cpu"] / CHUNK["ram"])
pq_go = sum(folder_size_gb(PARQUET / tr) for tr in USEFUL)
ratio = pq_go / csv_go
print(f"CSV {csv_go:.2f} Go -> Parquet {pq_go:.2f} Go (x{ratio:.2f})")
print(f"Estimation pour tout le dataset : {plan.csv_go.sum() * ratio:.0f} Go de Parquet")

from src.ingestion.ingest import SCHEMA
csv_paths = [(CSV_DIR / r / "ata_write.csv").as_posix() for r in runs.rel]
for name, df in [("CSV", spark.read.csv(csv_paths, schema=SCHEMA)), ("Parquet", ata)]:
    t = time.time()
    df.agg(F.avg("entropy"), F.count("*")).collect()
    print(f"{name:8s}: {time.time() - t:.1f} s")

# %% Cellule 7 : stats par exécution (lues depuis le Parquet) -> reports/run_stats.csv
stats = None
for tr in USEFUL:
    s = (spark.read.parquet((PARQUET / tr).as_posix())
         .groupBy("rel").agg(F.count("*").alias(f"{tr}_n"), F.max("t").alias(f"{tr}_duree_s"),
                             F.avg("entropy").alias(f"{tr}_entropie")).toPandas())
    stats = s if stats is None else stats.merge(s, on="rel", how="outer")
stats = inv[["rel", "category", "cpu", "ram", "sample", "family", "label"]].merge(stats, on="rel")
stats.to_csv(REPORTS / "run_stats.csv", index=False)
stats.groupby("sample")[["ata_write_n", "ata_write_duree_s", "ata_write_entropie"]].median().round(2)

# %% Cellule 8 : les autres morceaux (à lancer quand le premier est validé)
# Pour chaque morceau : extraire -> ingérer -> supprimer les CSV si DELETE_CSV.
DELETE_CSV = True       # garde False si tu veux garder les CSV
MIN_FREE_GO = 30        # on s'arrête s'il reste moins que ça sur le disque

done_file = PARQUET / "_chunks_done.txt"
done = set(done_file.read_text().split()) if done_file.exists() else set()
for _, c in plan.iterrows():
    key = f"{c.category}/{c.cpu}/{c.ram}"
    if key in done:
        continue
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < c.csv_go + MIN_FREE_GO:
        print(f"Stop : {free:.0f} Go libres, {key} demande {c.csv_go} Go"); break
    print(f"== {key} ({c.runs} exécutions, {c.csv_go} Go)")
    part = inv[(inv.category == c.category) & (inv.cpu == c.cpu) & (inv.ram == c.ram)]
    extract_subset(ZIP, CSV_DIR, part)
    ingest_chunk(spark, CSV_DIR, part, PARQUET, USEFUL)
    if DELETE_CSV:
        shutil.rmtree(CSV_DIR / c.category / c.cpu / c.ram, ignore_errors=True)
    with open(done_file, "a") as f:
        f.write(key + "\n")
print("Parquet total :", round(sum(folder_size_gb(PARQUET / tr) for tr in USEFUL), 2), "Go")

# %%
for tr in USEFUL:
    df = spark.read.parquet((PARQUET / tr).as_posix())
    print(tr, df.select("rel").distinct().count())
# %%
ar = {r.rel for r in spark.read.parquet((PARQUET / "ata_read").as_posix()).select("rel").distinct().collect()}
miss = inv[~inv.rel.isin(ar)]
display(miss[["rel", "label", "ata_read_mb"]])
# %%
