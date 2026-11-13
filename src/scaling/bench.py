"""Tâche 7 : mesure du passage à l'échelle de Spark (temps vs cœurs, partitions, taille des données).

Charge de travail = la vraie étape de la tâche 5 (features par fenêtre d'une trace),
écrite vers le format « noop » : Spark calcule tout mais n'écrit rien sur le disque,
donc on mesure le calcul et pas la vitesse du disque.
"""
import time

import pandas as pd
from pyspark.sql import SparkSession, functions as F

from src.features.windows import window_features


def make_spark(cores: int, shuffle_partitions: int, driver_memory="4g") -> SparkSession:
    """Redémarre Spark avec `cores` cœurs (local[cores])."""
    active = SparkSession.getActiveSession()
    if active is not None:
        active.stop()
    spark = (SparkSession.builder.appName(f"scaling-{cores}c")
             .master(f"local[{cores}]")
             .config("spark.driver.memory", driver_memory)
             .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def run_once(spark, parquet_root, offs, configs, trace="ata_write", W=1.0) -> float:
    """Calcule les features d'une trace pour une liste de configurations (category, cpu, ram).
    Renvoie le temps en secondes."""
    t = time.time()
    df = None
    for cat, cpu, ram in configs:
        f = window_features(spark, parquet_root, trace, W, offs,
                            subset=dict(category=cat, cpu=cpu, ram=ram))
        df = f if df is None else df.unionByName(f)
    df.write.format("noop").mode("overwrite").save()
    return time.time() - t


def bench(parquet_root, offs, configs, cores_list, shuffle_list, repeats=2, **kw) -> pd.DataFrame:
    """Toutes les combinaisons (cœurs, partitions) ; une première exécution « à blanc » par session
    (chauffe de la JVM) n'est pas comptée. Renvoie une ligne par mesure."""
    rows = []
    for cores in cores_list:
        for sp in shuffle_list:
            spark = make_spark(cores, sp)
            run_once(spark, parquet_root, offs, configs[:1], **kw)        # chauffe
            for r in range(repeats):
                sec = run_once(spark, parquet_root, offs, configs, **kw)
                rows.append(dict(cores=cores, shuffle_partitions=sp, n_configs=len(configs),
                                 repeat=r, seconds=round(sec, 2)))
                print(f"cœurs={cores:2d} partitions={sp:4d} configs={len(configs)} -> {sec:6.1f} s")
    return pd.DataFrame(rows)
