"""Création d'une SparkSession commune à tout le projet."""
from pyspark.sql import SparkSession

from src.utils.config import load_config


def get_spark() -> SparkSession:
    cfg = load_config()["spark"]
    return (
        SparkSession.builder.appName(cfg["app_name"])
        .master(cfg["master"])
        .config("spark.driver.memory", cfg["driver_memory"])
        .config("spark.sql.shuffle.partitions", cfg["shuffle_partitions"])
        .getOrCreate()
    )
