"""Test de fumée : vérifie que Spark démarre et sait écrire/relire du Parquet."""
from src.utils.spark_session import get_spark


def test_spark_parquet_roundtrip(tmp_path):
    spark = get_spark()
    df = spark.createDataFrame(
        [("conti", 1, 0.91), ("firefox", 0, 0.12)], ["sample", "label", "entropy"]
    )
    out = str(tmp_path / "smoke.parquet")
    df.write.mode("overwrite").parquet(out)
    back = spark.read.parquet(out)
    assert back.count() == 2
    assert set(back.columns) == {"sample", "label", "entropy"}
    spark.stop()
