import numpy as np

from src.evaluation.online import auc, roc_curve, pr_curve, first_alarm, choose_threshold, online_summary


def test_roc_pr_perfect_and_random():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.8, 0.9])
    fpr, tpr = roc_curve(y, p)
    assert auc(fpr, tpr) == 1.0
    rec, prec = pr_curve(y, p)
    assert rec[-1] == 1.0 and prec[2] == 1.0


def test_first_alarm_and_threshold():
    T = np.array([5, 10, 15, 20])
    S = np.array([[0.1, 0.6, 0.9, np.nan],     # ransomware : alerte à 10 s (k=1), 15 s (k=2)
                  [0.1, 0.2, 0.3, 0.4]])       # bénin : jamais
    assert first_alarm(T, S, 0.5).tolist()[0] == 10
    assert first_alarm(T, S, 0.5, k=2).tolist()[0] == 15
    assert np.isnan(first_alarm(T, S, 0.5)[1])
    thr = choose_threshold(T, S, np.array([1, 0]), target_far=0.0)
    assert 0.4 < thr <= 0.6
    r = online_summary(T, S, np.array([1, 0]), np.array([15, 20]), 0.5)
    assert r["taux_detection"] == 1 and r["fausses_alertes"] == 0 and r["temps_median_s"] == 10


def test_online_scores_shape():
    import torch
    from src.models.deep import make_model
    from src.evaluation.online import online_scores
    X = np.random.default_rng(0).normal(size=(3, 12, 2)).astype(np.float32)
    lengths = np.array([12, 7, 3])
    torch.manual_seed(0)
    Tt, S = online_scores(make_model("LSTM", 2).eval(), X, lengths, [0, 1, 2], step=5)
    assert Tt.tolist() == [5, 10, 15]
    assert not np.isnan(S[0]).any()                       # 12 fenêtres : 5, 10, 12
    assert np.isnan(S[1, 2]) and not np.isnan(S[1, 1])    # 7 fenêtres : 5, 7
    assert not np.isnan(S[2, 0]) and np.isnan(S[2, 1])    # 3 fenêtres : 3
