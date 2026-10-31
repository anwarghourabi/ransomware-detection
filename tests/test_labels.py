import pandas as pd

from src.ingestion.ingest import add_labels, family_of, label_of


def test_labels():
    assert label_of("Conti") == 1 and family_of("Conti") == "Conti"
    assert label_of("AESCrypt") == 0 and family_of("AESCrypt") == "AESCrypt"
    assert label_of("AESCrypt_LockBit") == 1 and family_of("AESCrypt_LockBit") == "LockBit"
    inv = add_labels(pd.DataFrame({"sample": ["Ryuk", "Zip"]}))
    assert inv["label"].tolist() == [1, 0]
