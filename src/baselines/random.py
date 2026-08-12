import numpy as np


def random_ranking(n_sensors: int, *, seed: int = 42) -> np.ndarray:
    """Complete uninformed permutation; nested subsets are its prefixes."""
    return np.random.default_rng(seed).permutation(int(n_sensors))


def paper_random_rankings(n_sensors: int, *, base_seed: int = 42, repetitions: int = 5):
    return [random_ranking(n_sensors, seed=base_seed + 50_000 + rep) for rep in range(repetitions)]
