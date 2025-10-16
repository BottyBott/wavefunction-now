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


def density_matrix_probabilities(
    rho: np.ndarray,
    effects: np.ndarray | list[np.ndarray],
) -> np.ndarray:
    r"""Return outcome probabilities ``Tr(E_i \rho)`` for a POVM effect set."""

    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix")
    dim = rho.shape[0]

    trace = float(np.real(np.trace(rho)))
    if trace <= 0.0:
        raise ValueError("rho must have positive trace")
    if not np.isclose(trace, 1.0, atol=1e-6):
        rho = rho / trace

    effects_arr = np.asarray(effects, dtype=complex)
    if effects_arr.ndim == 3:
        if effects_arr.shape[1:] != (dim, dim):
            raise ValueError("effects must match the dimension of rho")
        effect_mats = effects_arr
    elif effects_arr.ndim == 2 and effects_arr.shape[1] == dim:
        effect_mats = np.array([np.diag(mask) for mask in effects_arr], dtype=complex)
    else:
        raise ValueError("effects must be a stack of matrices or diagonal masks")

    probabilities = np.real(np.einsum("aij,ji->a", effect_mats, rho))
    if np.any(probabilities < -1e-8):
        raise ValueError("probabilities became negative; check POVM definition")
    probabilities = np.clip(probabilities, 0.0, None)
    if probabilities.sum() <= 0.0:
        raise ValueError("POVM produced zero total probability")
    return probabilities


def apply_detector_response(
    probabilities: np.ndarray,
    efficiency: float = 1.0,
    dark_count: float = 0.0,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Adjust ideal probabilities using a simple detector model.

    Parameters
    ----------
    probabilities:
        Idealised outcome probabilities (sum to one).
    efficiency:
        Fraction of true events recorded by each detector. Values below one
        model missed clicks; the output distribution is renormalised over the
        detected events.
    dark_count:
        Relative weight of spurious detector clicks. Set to zero for ideal
        detectors. Larger values inject additional uniform (or user-supplied)
        background counts that are mixed with the valid events before
        renormalisation.
    background:
        Optional background distribution for dark counts. Must be the same
        length as ``probabilities`` and non-negative. Defaults to a uniform
        background.
    """

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if np.any(probs < 0):
        raise ValueError("probabilities must be non-negative")
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError("probabilities must sum to one")

    if not (0.0 <= efficiency <= 1.0):
        raise ValueError("efficiency must lie between 0 and 1")
    if dark_count < 0.0:
        raise ValueError("dark_count must be non-negative")

    detected = efficiency * probs
    if background is None:
        background = np.ones_like(probs) / probs.size
    else:
        background = np.asarray(background, dtype=float)
        if background.shape != probs.shape:
            raise ValueError("background distribution must match probabilities shape")
        if np.any(background < 0):
            raise ValueError("background probabilities must be non-negative")
        norm = background.sum()
        if norm <= 0:
            raise ValueError("background distribution must have positive weight")
        background = background / norm

    detected = detected + dark_count * background
    total = detected.sum()
    if total <= 0.0:
        raise ValueError("detector response produced zero probability mass")
    return detected / total
