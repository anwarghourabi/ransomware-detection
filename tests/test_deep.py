import numpy as np
import torch

from src.models.deep import make_model, predict_proba


def test_models_ignore_padding():
    X = np.random.default_rng(0).normal(size=(2, 10, 3)).astype(np.float32)
    lengths = np.array([10, 4])
    for name in ["CNN", "LSTM", "GRU", "MHSA-LSTM"]:
        torch.manual_seed(0)
        m = make_model(name, 3).eval()
        p1 = predict_proba(m, X, lengths, [0, 1])
        X2 = X.copy(); X2[1, 4:] = 99.0          # changer le padding ne doit rien changer
        p2 = predict_proba(m, X2, lengths, [0, 1])
        assert p1.shape == (2,) and abs(p1[1] - p2[1]) < 1e-5, name


def test_attention_is_causal_and_ig():
    from src.models.deep import info_gain, ig_weights, attention_profile
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 12, 3)).astype(np.float32)
    lengths = rng.integers(3, 13, size=40)
    y = np.array([0, 1] * 20)
    X[y == 1, :, 0] += 3.0                       # seule la feature 0 sépare les classes
    ig = info_gain(X, lengths, y, np.arange(40))
    assert ig.argmax() == 0 and ig[0] > 0.9
    w = ig_weights(ig)
    assert w.shape == (3,) and abs(w.mean() - 1) < 0.5
    torch.manual_seed(0)
    m = make_model("MHSA-LSTM-IG", 3, feat_weights=w).eval()
    prof = attention_profile(m, X, lengths, np.arange(40))
    assert prof.shape == (40, 20) and np.allclose(prof.sum(1), 1, atol=1e-4)
    # causalité : la prédiction sur un préfixe ne dépend pas de ce qui vient après
    p_full = predict_proba(m, X, np.full(40, 12), [0])
    X2 = X.copy(); X2[0, 6:] = 50.0
    p_pref1 = predict_proba(m, X, np.full(40, 6), [0])
    p_pref2 = predict_proba(m, X2, np.full(40, 6), [0])
    assert abs(p_pref1[0] - p_pref2[0]) < 1e-5 and p_full.shape == (1,)


def test_multi_prefix_training_runs():
    from src.models.deep import train
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 20, 3)).astype(np.float32)
    lengths = rng.integers(5, 21, size=30)
    y = (np.arange(30) % 2).astype(np.float32)
    m = train("LSTM", X, lengths, y, np.arange(20), np.arange(20, 30), epochs=2,
              verbose=False, prefix_fracs=[0.1, 0.5, 1.0])
    p = predict_proba(m, X, np.maximum(1, lengths // 2), np.arange(30))
    assert p.shape == (30,) and np.all((p >= 0) & (p <= 1))
