"""Tâche 5 : caractéristiques par fenêtre temporelle (mémoire et stockage).

Entrée  : data/processed/parquet/<trace>/   (tâche 4, une ligne par événement)
Sortie  : data/processed/features/w=<W>/    (une ligne par exécution et par fenêtre)

Fenêtre n° w d'une exécution = les événements avec  w*W <= t_aligné < (w+1)*W.
"""
import zipfile
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F

from src.ingestion.ransmap_zip import _parse

PREFIX = {"ata_read": "ar", "ata_write": "aw", "mem_read": "mr", "mem_write": "mw"}
HIGH_ENTROPY = 0.9          # entropie normalisée (0 à 1) : >= 0.9 ~ données chiffrées/compressées
META = ["rel", "category", "cpu", "ram", "family", "sample", "label"]


# ---------- 1. alignement des 4 traces d'une exécution ----------
def first_ts_from_zip(zip_path, traces, n_lines=1000) -> pd.DataFrame:
    """Lit seulement les n premières lignes de chaque CSV dans le zip (rapide)
    et renvoie le plus petit horodatage : (rel, trace, first_ts)."""
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            p = _parse(info.filename)
            if p is None or p[5] not in traces:
                continue
            best = None
            with zf.open(info) as f:
                for i, line in enumerate(f):
                    if i >= n_lines:
                        break
                    parts = line.decode(errors="ignore").strip().split(",")
                    if len(parts) < 2 or not parts[0].strip().isdigit():
                        continue
                    ts = int(parts[0]) + int(parts[1]) / 1e9
                    best = ts if best is None else min(best, ts)
            if best is not None:
                rows.append(("/".join(p[:5]), p[5], best))
    return pd.DataFrame(rows, columns=["rel", "trace", "first_ts"])


def offsets(first_ts: pd.DataFrame) -> pd.DataFrame:
    """Décalage de chaque trace par rapport au début de l'exécution (la trace qui commence en premier).
    En tâche 4, t part de 0 pour CHAQUE trace ; t_aligné = t + offset les remet sur la même horloge."""
    start = first_ts.groupby("rel")["first_ts"].transform("min")
    return first_ts.assign(offset=(first_ts["first_ts"] - start).round(6))[["rel", "trace", "offset"]]


# ---------- 2. caractéristiques d'une trace par fenêtre ----------
def read_trace(spark, parquet_root, trace, subset=None) -> DataFrame:
    """subset = dict(category=..., cpu=..., ram=...) pour ne lire qu'une partie (rapide)."""
    df = spark.read.parquet((Path(parquet_root) / trace).as_posix())
    for k, v in (subset or {}).items():
        df = df.where(F.col(k) == v)
    return df


def window_features(spark: SparkSession, parquet_root, trace: str, W: float,
                    offs: pd.DataFrame, subset=None) -> DataFrame:
    p = PREFIX[trace]
    off = spark.createDataFrame(offs[offs.trace == trace][["rel", "offset"]])
    df = (read_trace(spark, parquet_root, trace, subset)
          .select("rel", "t", "address", "size", "entropy")
          .join(F.broadcast(off), "rel", "left")
          .withColumn("w", F.floor((F.col("t") + F.coalesce("offset", F.lit(0.0))) / W).cast("int")))
    return df.groupBy("rel", "w").agg(
        F.count("*").alias(f"{p}_n"),                                     # nombre d'accès
        F.sum("size").alias(f"{p}_bytes"),                                # volume (octets)
        F.approx_count_distinct("address").alias(f"{p}_addr_uniq"),       # adresses distinctes
        F.stddev_pop("address").alias(f"{p}_addr_std"),                   # dispersion des accès
        F.avg("entropy").alias(f"{p}_ent_mean"),                          # entropie moyenne
        F.max("entropy").alias(f"{p}_ent_max"),
        F.avg((F.col("entropy") >= HIGH_ENTROPY).cast("double")).alias(f"{p}_ent_hi"),  # part chiffrée
    )


def build_features(spark, parquet_root, runs: pd.DataFrame, offs: pd.DataFrame, W: float,
                   traces=tuple(PREFIX), subset=None) -> DataFrame:
    """Une ligne par (exécution, fenêtre) avec les 4 traces côte à côte + ratios lecture/écriture.
    Les fenêtres sans aucun événement sont absentes ici ; elles seront remplies par 0 en tâche 6."""
    feats = None
    for tr in traces:
        f = window_features(spark, parquet_root, tr, W, offs, subset)
        feats = f if feats is None else feats.join(f, ["rel", "w"], "full_outer")
    feats = feats.fillna(0)
    feats = (feats
             .withColumn("disk_read_ratio", F.col("ar_n") / (F.col("ar_n") + F.col("aw_n") + F.lit(1e-9)))
             .withColumn("mem_read_ratio", F.col("mr_n") / (F.col("mr_n") + F.col("mw_n") + F.lit(1e-9))))
    meta = spark.createDataFrame(runs[META].astype({"label": "int32"}))
    return feats.join(F.broadcast(meta), "rel").withColumn("label", F.col("label").cast("int")).select(*META, "w", *[c for c in feats.columns if c not in ("rel", "w")])


def feature_columns(df) -> list:
    return [c for c in df.columns if c not in META and c != "w"]
