import numpy as np

from baselines import grouped_mrmr, loro_ranking, permutation_ranking, qr_response_diversity, random_ranking


def _complete(ranking, m):
    assert sorted(np.asarray(ranking).tolist()) == list(range(m))


def test_simple_baselines_return_complete_rankings():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 5, 3))
    y = x[:, 2, 0] + rng.normal(scale=0.01, size=40)
    _complete(random_ranking(5), 5)
    _complete(qr_response_diversity(x), 5)
    rank = grouped_mrmr(x, y)
    _complete(rank, 5)
    assert rank[0] == 2


def test_model_based_baselines_return_complete_rankings():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 4, 2))
    y = x[:, 1, 0] + rng.normal(scale=0.05, size=50)
    _complete(permutation_ranking(x[:35], y[:35], x[35:], y[35:], repeats=2), 4)
    _complete(permutation_ranking(x[:35], y[:35], x[35:], y[35:], mode="high_error", repeats=2), 4)
    _complete(loro_ranking(x[:35], y[:35], x[35:], y[35:]), 4)
