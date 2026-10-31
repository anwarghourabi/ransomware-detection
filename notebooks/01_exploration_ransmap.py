# %% [markdown]
# # Exploration de RanSMAP en local (tâche 3)
# Exécuter cellule par cellule dans VS Code (« Run Cell » au-dessus de chaque bloc).

# %% Cellule 1 : chemins et imports
import os
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
os.environ["PYSPARK_PYTHON"] = sys.executable          # Spark utilise le Python du venv
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import matplotlib.pyplot as plt
import pandas as pd

from src.ingestion.ransmap_zip import TRACES, USEFUL, extract_subset, inventory_from_zip

ZIP = ROOT / "data" / "raw" / "archive.zip"
SUBSET_DIR = ROOT / "data" / "raw" / "ransmap"         # fichiers extraits (sous-ensemble)
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
print("Archive trouvée :", ZIP.exists(), "|", round(ZIP.stat().st_size / 1e9, 1), "Go")

# %% Cellule 2 : inventaire directement depuis le zip (rien n'est décompressé)
inv = inventory_from_zip(ZIP)
print(len(inv), "exécutions ;", round(inv["total_mb"].sum() / 1e3, 1), "Go une fois décompressé")
inv.groupby("category").agg(executions=("rel", "size"),
                            go=("total_mb", lambda s: round(s.sum() / 1e3, 1)))
# Attendu : extra 360, mix 100, original 1440, variants 70 ; total 1970 et 172 Go

# %% Cellule 3 : échantillons et configurations matérielles
print(sorted(inv["sample"].unique()))
display(pd.crosstab([inv["cpu"], inv["ram"]], inv["category"]))
display(pd.crosstab(inv["sample"], inv["category"]))

# %% Cellule 4 : étiquetage (1 = ransomware, 0 = bénin)
RANSOMWARE = ["Conti", "Darkside", "LockBit", "REvil", "Ryuk", "WannaCry"]

def family(sample):
    parts = sample.split("_")
    rans = [p for p in parts if p in RANSOMWARE]
    return rans[0] if rans else parts[0]

inv["label"] = inv["sample"].map(lambda s: int(any(p in RANSOMWARE for p in s.split("_"))))
inv["family"] = inv["sample"].map(family)
inv.to_csv(REPORTS / "inventaire_ransmap.csv", index=False)
display(pd.crosstab(inv["family"], inv["label"]))
inv.groupby(["category", "label"]).size()

# %% Cellule 5 : volumes par type de fichier
display(inv[[f"{t}_mb" for t in TRACES]].describe().T.round(1))
inv[inv.category == "original"].groupby("sample")["total_mb"].median().sort_values().plot.barh(figsize=(7, 5))
plt.xlabel("Taille médiane par exécution (Mo)"); plt.tight_layout()
plt.savefig(FIG / "fig_volumes.png", dpi=150); plt.show()

# %% Cellule 6 : extraire UNE configuration (120 exécutions, environ 10 Go) — à faire une seule fois
CONFIG = dict(category="original", cpu="i3-gen12", ram="ddr4-2133-16g")
config_runs = inv[(inv.category == CONFIG["category"]) & (inv.cpu == CONFIG["cpu"]) & (inv.ram == CONFIG["ram"])]
print(len(config_runs), "exécutions ;",
      round(config_runs[[f"{t}_mb" for t in USEFUL]].sum().sum() / 1e3, 1), "Go à extraire")
extract_subset(ZIP, SUBSET_DIR, config_runs)   # relancer ne réextrait pas ce qui existe déjà
config_runs = config_runs.assign(run_dir=config_runs["rel"].map(lambda r: str(SUBSET_DIR / r)))

# %% Cellule 7 : démarrer Spark
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, LongType, StructField, StructType

spark = (SparkSession.builder.master("local[*]").appName("ransmap")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

SCHEMA = StructType([
    StructField("sec", LongType()), StructField("ns", LongType()),
    StructField("address", LongType()),   # LBA (disque) ou GPA (mémoire)
    StructField("size", LongType()),
    StructField("entropy", DoubleType()),  # -1 si non mesurée
    StructField("type", LongType()),
])

def read_many(paths):
    """Lit plusieurs CSV en un seul DataFrame, avec le nom de l'exécution."""
    return (spark.read.csv(paths, schema=SCHEMA, header=False,
                           ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)
            .where(F.col("sec").isNotNull())
            .withColumn("ts", F.col("sec") + F.col("ns") / 1e9)
            .withColumn("run_name", F.element_at(F.split(F.input_file_name(), "/"), -2)))

spark.read.csv(str(Path(config_runs.run_dir.iloc[0]) / "ata_write.csv"), schema=SCHEMA,
               ignoreLeadingWhiteSpace=True).show(5)

# %% Cellule 8 : statistiques (3 exécutions par échantillon)
subset = config_runs.groupby("sample").head(3)
print(len(subset), "exécutions")

def stats_for(t):
    df = read_many([str(Path(d) / f"{t}.csv") for d in subset["run_dir"]])
    return df.groupBy("run_name").agg(
        F.count("*").alias(f"{t}_n"),
        (F.max("ts") - F.min("ts")).alias(f"{t}_duree_s"),
        F.avg(F.when(F.col("entropy") >= 0, F.col("entropy"))).alias(f"{t}_entropie"),
        F.approx_count_distinct("address").alias(f"{t}_adresses"),
    ).toPandas()

stats = subset[["run_name", "sample", "family", "label"]]
for t in ["ata_write", "ata_read"]:
    stats = stats.merge(stats_for(t), on="run_name", how="left")
    print(t, "terminé")

stats.to_csv(REPORTS / "stats_i3_ddr4-2133-16g.csv", index=False)
stats.groupby("sample").median(numeric_only=True).round(3)

# %% Cellule 9 : durées et entropie (rouge = ransomware)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
stats.boxplot(column="ata_write_duree_s", by="sample", rot=90, ax=ax[0])
ax[0].set_title("Durée des exécutions (s)")
med = stats.groupby("sample").agg(e=("ata_write_entropie", "median"), label=("label", "first")).sort_values("e")
med["e"].plot.barh(ax=ax[1], color=med["label"].map({1: "tab:red", 0: "tab:blue"}))
ax[1].set_title("Entropie moyenne des écritures disque (rouge = ransomware)")
plt.suptitle(""); plt.tight_layout()
plt.savefig(FIG / "fig_durees_entropie.png", dpi=150); plt.show()

# %% Cellule 10 : un ransomware et un programme bénin dans le temps
def timeline(run_dir):
    df = read_many([str(Path(run_dir) / "ata_write.csv")])
    tl = (df.groupBy(F.floor("ts").alias("sec_bin"))
            .agg(F.count("*").alias("n"), F.avg("entropy").alias("entropie"))
            .orderBy("sec_bin").toPandas())
    tl["t"] = tl["sec_bin"] - tl["sec_bin"].min()
    return tl

r = subset[subset["sample"] == "Conti"].iloc[0]
b = subset[subset["sample"] == "AESCrypt"].iloc[0]
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for row, name in [(r, "ransomware (Conti)"), (b, "bénin (AESCrypt)")]:
    tl = timeline(row["run_dir"])
    ax[0].plot(tl.t, tl.n, label=name)
    ax[1].plot(tl.t, tl.entropie, label=name)
ax[0].set_ylabel("écritures disque / s"); ax[1].set_ylabel("entropie moyenne")
ax[1].set_xlabel("secondes depuis le début"); ax[0].legend()
plt.tight_layout(); plt.savefig(FIG / "fig_timeline.png", dpi=150); plt.show()

# %%
