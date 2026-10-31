"""Tâche 4 : ingestion PySpark de RanSMAP -> Parquet étiqueté.

Chaque fichier de traces CSV (sans en-tête) devient des lignes Parquet avec :
    rel, run_name, sample, label, t, address, size, entropy, type
partitionnées par trace / category / cpu / ram / family.

On ingère une configuration matérielle à la fois (un « morceau ») pour rester dans
la place disponible : extraire -> ingérer -> (optionnel) supprimer les CSV.
"""
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import DoubleType, LongType, StructField, StructType

RANSOMWARE = ["Conti", "Darkside", "LockBit", "REvil", "Ryuk", "WannaCry"]
PARTITIONS = ["category", "cpu", "ram", "family"]

SCHEMA = StructType([
    StructField("sec", LongType()), StructField("ns", LongType()),
    StructField("address", LongType()),    # LBA (disque) ou GPA (mémoire)
    StructField("size", LongType()),
    StructField("entropy", DoubleType()),  # -1 si non mesurée
    StructField("type", LongType()),
])


# ---------- étiquetage (pandas, sans Spark) ----------
def family_of(sample: str) -> str:
    """'Conti' -> 'Conti' ; 'AESCrypt_Conti' (mix) -> 'Conti' ; 'Zip' -> 'Zip'."""
    parts = sample.split("_")
    rans = [p for p in parts if p in RANSOMWARE]
    return rans[0] if rans else parts[0]


def label_of(sample: str) -> int:
    """1 si l'échantillon contient un ransomware (mix et variants compris), sinon 0."""
    return int(any(p in RANSOMWARE for p in sample.split("_")))


def add_labels(inv: pd.DataFrame) -> pd.DataFrame:
    inv = inv.copy()
    inv["family"] = inv["sample"].map(family_of)
    inv["label"] = inv["sample"].map(label_of)
    return inv


def chunks(inv: pd.DataFrame, traces) -> pd.DataFrame:
    """Une ligne par morceau (category, cpu, ram) avec le volume CSV à extraire."""
    mb = inv[[f"{t}_mb" for t in traces]].sum(axis=1)
    return (inv.assign(csv_mb=mb)
            .groupby(["category", "cpu", "ram"], as_index=False)
            .agg(runs=("rel", "size"), csv_go=("csv_mb", lambda s: round(s.sum() / 1e3, 1)))
            .sort_values(["category", "cpu", "ram"]).reset_index(drop=True))


def folder_size_gb(path) -> float:
    """Taille d'un dossier en Go (remplace `du`, marche sous Windows)."""
    p = Path(path)
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


# ---------- ingestion (Spark) ----------
def read_trace(spark: SparkSession, csv_root, runs: pd.DataFrame, trace: str) -> DataFrame:
    """Lit <csv_root>/<rel>/<trace>.csv pour toutes les exécutions de `runs` en un seul job."""
    csv_root = Path(csv_root)
    paths = [(csv_root / r / f"{trace}.csv") for r in runs["rel"]]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"  ! {len(missing)} fichiers {trace}.csv manquants (ignorés), ex. {missing[0]}")
    paths = [p.as_posix() for p in paths if p.exists()]
    if not paths:
        return None

    meta = spark.createDataFrame(
        runs[["rel", "category", "cpu", "ram", "sample", "run_name", "family", "label"]]
        .astype({"label": "int32"}))

    parts = F.split(F.regexp_replace(F.input_file_name(), r"\\", "/"), "/")
    rel = F.concat_ws("/", *[F.element_at(parts, i) for i in range(-6, -1)])

    df = (spark.read.csv(paths, schema=SCHEMA, header=False,
                         ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)
          .where(F.col("sec").isNotNull())
          .withColumn("rel", rel))

    # t = secondes depuis le premier événement de l'exécution (pour couper à 10/25/50/75 %)
    # calculé avec sec et ns séparés pour garder la précision à la nanoseconde
    t0 = (df.groupBy("rel").agg(F.min(F.struct("sec", "ns")).alias("m"))
          .select("rel", F.col("m.sec").alias("sec0"), F.col("m.ns").alias("ns0")))
    return (df.join(F.broadcast(t0), "rel")
            .join(F.broadcast(meta), "rel")
            .select("rel", *PARTITIONS[:-1], "family", "sample", "run_name", F.col("label").cast("int").alias("label"),
                    ((F.col("sec") - F.col("sec0")) + (F.col("ns") - F.col("ns0")) / 1e9).alias("t"),
                    "address", F.col("size").cast("int").alias("size"),
                    F.when(F.col("entropy") >= 0, F.col("entropy")).cast("float").alias("entropy"),
                    F.col("type").cast("int").alias("type")))


def ingest_chunk(spark: SparkSession, csv_root, runs: pd.DataFrame, out_root, traces) -> None:
    """Écrit out_root/<trace>/category=../cpu=../ram=../family=../part-*.parquet.
    Réécrire un morceau ne remplace que ses propres partitions (mode dynamique)."""
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    for trace in traces:
        df = read_trace(spark, csv_root, runs, trace)
        if df is None:
            print(f"  {trace} : aucun fichier, ignoré")
            continue
        out = (Path(out_root) / trace).as_posix()
        df.write.mode("overwrite").partitionBy(*PARTITIONS).parquet(out)
        print(f"  {trace} -> {out}")
