import numpy as np
import pandas as pd

from src.sequences.build import make_splits, prefix, prefix_lengths, to_sequences


def test_prefix():
    lengths = np.array([200, 10])
    assert prefix_lengths(lengths, 0.1).tolist() == [20, 1]
    assert prefix_lengths(lengths, 1.0).tolist() == [200, 10]
    X = np.ones((2, 200, 3), dtype=np.float32)
    Xp, lp = prefix(X, lengths, 0.25)
    assert lp.tolist() == [50, 3] and Xp[0, :50].sum() == 150 and Xp[0, 50:].sum() == 0


def test_sequences_and_splits():
    fe = pd.DataFrame({"rel": ["a", "a", "b"], "category": "original", "cpu": "c", "ram": "r",
                       "family": ["F", "F", "G"], "sample": ["F", "F", "G"], "label": [1, 1, 0],
                       "w": [0, 3, 1], "x": [1.0, 2.0, 5.0]})
    X, lengths, runs = to_sequences(fe, ["x"])
    assert X.shape == (2, 4, 1) and lengths.tolist() == [4, 2]
    assert X[0, :, 0].tolist() == [1, 0, 0, 2] and X[1, :, 0].tolist() == [0, 5, 0, 0]
    runs = pd.DataFrame({"category": "original", "sample": ["S"] * 10, "cpu": "c", "ram": "r"})
    s = make_splits(runs)
    assert (s == "train").sum() == 7 and (s == "val").sum() == 1 and (s == "test").sum() == 2
