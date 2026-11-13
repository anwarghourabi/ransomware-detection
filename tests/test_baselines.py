import numpy as np

from src.models.baselines import metrics, prefix_aggregates


def test_prefix_aggregates():
    X = np.zeros((1, 10, 1), dtype=np.float32)
    X[0, :, 0] = np.arange(10)
    a = prefix_aggregates(X, np.array([10]), 0.5, ["x"])   # 5 premières fenêtres : 0..4
    assert a["x__mean"][0] == 2 and a["x__min"][0] == 0 and a["x__max"][0] == 4


def test_metrics():
    m = metrics([1, 1, 0, 0], [0.9, 0.2, 0.1, 0.7])
    assert m["tp"] == 1 and m["fp"] == 1 and m["recall"] == 0.5 and m["fpr"] == 0.5
    assert metrics([1, 0], [0.9, 0.1])["auc_roc"] == 1.0
