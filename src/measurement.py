"""Measurement utilities: from \(\psi\) to event statistics.

We never observe \(\psi\) directly. Instead we observe localised detector
events whose frequencies follow \(|\psi|^2\). These helpers convert the
wave-function model into the empirical correlations we actually record.
"""

from __future__ import annotations

import numpy as np


def born_probability(psi: np.ndarray) -> np.ndarray:
    """Return normalised probabilities |psi|^2."""
    prob = np.abs(psi) ** 2
    total = prob.sum()
    if not np.isclose(total, 1.0):
        prob /= total
    return prob


def sample_measurements(prob: np.ndarray, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample measurement outcomes according to `prob`."""
    if rng is None:
        rng = np.random.default_rng()
    indices = np.arange(prob.size)
    return rng.choice(indices, size=size, p=prob)
