import numpy as np
import torch

from src.models.deep import make_model, predict_proba


def test_models_ignore_padding():
    X = np.random.default_rng(0).normal(size=(2, 10, 3)).astype(np.float32)
    lengths = np.array([10, 4])
    for name in ["CNN", "LSTM", "GRU"]:
        torch.manual_seed(0)
        m = make_model(name, 3).eval()
        p1 = predict_proba(m, X, lengths, [0, 1])
        X2 = X.copy(); X2[1, 4:] = 99.0          # changer le padding ne doit rien changer
        p2 = predict_proba(m, X2, lengths, [0, 1])
        assert p1.shape == (2,) and abs(p1[1] - p2[1]) < 1e-5, name
