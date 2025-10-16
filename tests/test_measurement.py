import numpy as np

from wavefunction_now.measurement import born_probability, sample_measurements


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
