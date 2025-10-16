r"""Measurement utilities: from \(\psi\) to event statistics.

We never observe \(\psi\) directly. Instead we observe localised detector
events whose frequencies follow \(|\psi|^2\). These helpers convert the
wave-function model into the empirical correlations we actually record.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def born_probability(psi: np.ndarray) -> np.ndarray:
    """Return normalised probabilities |psi|^2."""
    prob = np.abs(np.asarray(psi)) ** 2
    total = prob.sum()
    if total <= 0:
        raise ValueError("wave function has zero norm")
    if not np.isclose(total, 1.0):
        prob = prob / total
    return prob


def sample_measurements(prob: np.ndarray, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample measurement outcomes according to `prob`."""
    if size < 0:
        raise ValueError("sample size must be non-negative")
    if rng is None:
        rng = np.random.default_rng()
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 1:
        raise ValueError("probability array must be one-dimensional")
    if not np.isclose(prob.sum(), 1.0):
        raise ValueError("probabilities must sum to one")
    if np.any(prob < 0):
        raise ValueError("probabilities must be non-negative")
    indices = np.arange(prob.size)
    return rng.choice(indices, size=size, p=prob)


def projective_measurement(
    psi: np.ndarray,
    projectors: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Perform a projective measurement defined by diagonal projectors.

    Parameters
    ----------
    psi:
        State vector to be measured.
    projectors:
        Array of shape (n_projectors, len(psi)) containing boolean or float masks.
        Each mask defines which amplitudes survive the projection.

    Returns
    -------
    outcome:
        Selected projector index.
    post_state:
        Normalised post-measurement state.
    probabilities:
        Born probabilities for each outcome.
    """

    psi = np.asarray(psi, dtype=complex)
    projectors = np.asarray(projectors, dtype=float)
    if psi.ndim != 1:
        raise ValueError("psi must be a one-dimensional state vector")
    if projectors.ndim != 2 or projectors.shape[1] != psi.size:
        raise ValueError("projectors must be of shape (n_projectors, len(psi))")

    if rng is None:
        rng = np.random.default_rng()

    masked = projectors * psi
    probabilities = np.sum(np.abs(masked) ** 2, axis=1)
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("all projectors annihilate the state")
    probabilities = probabilities / total

    outcome = rng.choice(projectors.shape[0], p=probabilities)
    post_state = masked[outcome]
    norm = np.linalg.norm(post_state)
    if norm == 0:
        raise ValueError("selected projector annihilated the state")
    post_state = post_state / norm
    return int(outcome), post_state, probabilities


def chi_squared_gof(predicted: np.ndarray, observed_counts: np.ndarray) -> tuple[float, float]:
    """Return chi-squared statistic and p-value comparing predicted vs observed counts."""
    predicted = np.asarray(predicted, dtype=float)
    observed_counts = np.asarray(observed_counts, dtype=float)
    if predicted.ndim != 1 or observed_counts.ndim != 1:
        raise ValueError("predicted and observed_counts must be one-dimensional")
    if predicted.size != observed_counts.size:
        raise ValueError("arrays must have the same length")
    if np.any(predicted < 0) or np.any(observed_counts < 0):
        raise ValueError("probabilities and counts must be non-negative")
    total_prob = predicted.sum()
    if not np.isclose(total_prob, 1.0):
        raise ValueError("predicted probabilities must sum to one")
    total_counts = observed_counts.sum()
    if total_counts <= 0:
        raise ValueError("observed_counts must contain at least one event")

    expected = predicted * total_counts
    if np.any(expected == 0):
        raise ValueError("expected frequencies must be non-zero to compute chi-squared")

    statistic, p_value = stats.chisquare(observed_counts, expected)
    return float(statistic), float(p_value)


def ks_goodness_of_fit(predicted: np.ndarray, samples: np.ndarray) -> tuple[float, float]:
    """Return Kolmogorov-Smirnov statistic and p-value for discrete outcomes."""
    predicted = np.asarray(predicted, dtype=float)
    samples = np.asarray(samples)
    if predicted.ndim != 1:
        raise ValueError("predicted probabilities must be one-dimensional")
    if predicted.size == 0:
        raise ValueError("predicted probabilities cannot be empty")
    if np.any(predicted < 0):
        raise ValueError("predicted probabilities must be non-negative")
    total_prob = predicted.sum()
    if not np.isclose(total_prob, 1.0):
        raise ValueError("predicted probabilities must sum to one")
    if samples.ndim != 1:
        raise ValueError("samples must be a one-dimensional array")
    if samples.size == 0:
        raise ValueError("samples array must contain at least one value")
    if not np.issubdtype(samples.dtype, np.integer):
        raise ValueError("samples must be integer-valued indices")
    if np.any(samples < 0) or np.any(samples >= predicted.size):
        raise ValueError("samples contain indices outside the probability support")

    counts = np.bincount(samples, minlength=predicted.size).astype(float)
    n = counts.sum()
    if n <= 0:
        raise ValueError("samples array must contain at least one value")

    empirical_cdf = np.cumsum(counts) / n
    predicted_cdf = np.cumsum(predicted)
    d_statistic = np.max(np.abs(empirical_cdf - predicted_cdf))

    # Kolmogorov distribution survival function; conservative for discrete bins.
    p_value = stats.ksone.sf(d_statistic, n)
    return float(d_statistic), float(p_value)
