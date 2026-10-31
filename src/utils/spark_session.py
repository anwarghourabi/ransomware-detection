"""Création d'une SparkSession commune à tout le projet."""
import os
import sys

from pyspark.sql import SparkSession

from src.utils.config import load_config

# Spark doit utiliser le Python du venv (sinon « Python was not found » sous Windows)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
# Sous Windows, Spark a besoin de winutils.exe pour écrire des fichiers
if os.name == "nt" and "HADOOP_HOME" not in os.environ and os.path.exists(r"D:\hadoop\bin\winutils.exe"):
    os.environ["HADOOP_HOME"] = r"D:\hadoop"
    os.environ["PATH"] += r";D:\hadoop\bin"


def get_spark() -> SparkSession:
    cfg = load_config()["spark"]
    return (
        SparkSession.builder.appName(cfg["app_name"])
        .master(cfg["master"])
        .config("spark.driver.memory", cfg["driver_memory"])
        .config("spark.sql.shuffle.partitions", cfg["shuffle_partitions"])
        .getOrCreate()
    )