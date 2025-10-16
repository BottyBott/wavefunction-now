import numpy as np

from wavefunction_now.measurement import (
    born_probability,
    chi_squared_gof,
    ks_goodness_of_fit,
    projective_measurement,
    sample_measurements,
)


def test_born_probability_normalises():
    psi = np.array([1 + 0j, 1 + 0j])
    prob = born_probability(psi)
    assert np.allclose(prob, [0.5, 0.5])


def test_sampling_matches_probabilities():
    prob = np.array([0.2, 0.8])
    samples = sample_measurements(prob, size=10000, rng=np.random.default_rng(42))
    counts = np.bincount(samples, minlength=prob.size) / samples.size
    assert np.allclose(counts, prob, atol=0.01)


def test_sampling_respects_deterministic_cases():
    prob = np.array([0.0, 1.0])
    samples = sample_measurements(prob, size=1000, rng=np.random.default_rng(1))
    assert np.all(samples == 1)


def test_projective_measurement_collapses_state():
    psi = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    projectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    rng = np.random.default_rng(7)

    outcome, post_state, probabilities = projective_measurement(psi, projectors, rng=rng)
    assert np.isclose(np.linalg.norm(post_state), 1.0)
    assert np.allclose(probabilities, [0.5, 0.5])

    expected = np.zeros_like(psi)
    expected[outcome] = 1.0
    assert np.allclose(post_state, expected)


def test_chi_squared_gof_accepts_valid_samples():
    prob = np.array([0.3, 0.4, 0.3])
    rng = np.random.default_rng(123)
    samples = sample_measurements(prob, size=5000, rng=rng)
    counts = np.bincount(samples, minlength=prob.size)
    statistic, p_value = chi_squared_gof(prob, counts)
    assert statistic >= 0
    assert 0 <= p_value <= 1
    assert p_value > 0.05


def test_chi_squared_gof_detects_mismatch():
    prob = np.array([0.5, 0.3, 0.2])
    counts = np.array([100, 10, 890])
    statistic, p_value = chi_squared_gof(prob, counts)
    assert statistic > 0
    assert p_value < 1e-6


def test_ks_goodness_of_fit_accepts_valid_samples():
    prob = np.array([0.25, 0.5, 0.25])
    rng = np.random.default_rng(321)
    samples = sample_measurements(prob, size=4000, rng=rng)
    d_stat, p_value = ks_goodness_of_fit(prob, samples)
    assert 0 <= d_stat <= 1
    assert 0 <= p_value <= 1
    assert p_value > 0.05


def test_ks_goodness_of_fit_detects_mismatch():
    prob = np.array([0.1, 0.2, 0.7])
    samples = np.repeat(np.arange(prob.size), [900, 50, 50])
    d_stat, p_value = ks_goodness_of_fit(prob, samples)
    assert d_stat > 0.1
    assert p_value < 1e-6
