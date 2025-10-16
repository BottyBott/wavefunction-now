import numpy as np
import pytest

from wavefunction_now.solver import SplitStepSimulator
from wavefunction_now.measurement import born_probability


def test_gaussian_normalisation():
    sim = SplitStepSimulator(grid_points=512, length=20.0)
    psi = sim.gaussian_wavepacket(sim.x, x0=0.0, p0=0.0, sigma=1.0)
    prob = born_probability(psi)
    assert pytest.approx(prob.sum(), rel=1e-3) == 1.0


def test_free_evolution_norm_conserved():
    sim = SplitStepSimulator(grid_points=512, length=20.0)
    psi0 = sim.gaussian_wavepacket(sim.x, x0=-2.0, p0=2.0, sigma=1.0)
    potential = np.zeros_like(sim.x)
    psi_t = sim.evolve(psi0, potential, dt=0.01, steps=100)
    prob0 = born_probability(psi0)
    probt = born_probability(psi_t)
    assert pytest.approx(prob0.sum(), rel=1e-3) == pytest.approx(probt.sum(), rel=1e-3)


def test_position_projectors_produce_expected_probabilities():
    sim = SplitStepSimulator(grid_points=128, length=10.0)
    psi = sim.gaussian_wavepacket(sim.x, x0=-1.0, p0=0.0, sigma=0.7)
    bins = np.array([-5.0, 0.0, 5.0])
    bin_probabilities = sim.bin_probabilities(psi, bins)
    full_prob = born_probability(psi)

    manual_left = full_prob[(sim.x >= -5.0) & (sim.x < 0.0)].sum()
    manual_right = full_prob[(sim.x >= 0.0) & (sim.x <= 5.0)].sum()

    assert pytest.approx(bin_probabilities.sum(), rel=1e-3) == 1.0
    assert pytest.approx(bin_probabilities[0], rel=1e-3) == manual_left
    assert pytest.approx(bin_probabilities[1], rel=1e-3) == manual_right


def test_measure_position_collapses_to_selected_bin():
    sim = SplitStepSimulator(grid_points=128, length=10.0)
    psi = sim.gaussian_wavepacket(sim.x, x0=1.2, p0=0.0, sigma=0.6)
    bins = np.array([-5.0, 0.0, 5.0])
    rng = np.random.default_rng(99)
    outcome, post_state, probabilities = sim.measure_position(psi, bins, rng=rng)

    assert pytest.approx(probabilities, rel=1e-6) == sim.bin_probabilities(psi, bins)
    assert np.isclose(np.linalg.norm(post_state), 1.0)

    projectors = sim.position_projectors(bins)
    support = projectors[outcome].astype(bool)
    assert np.allclose(post_state[~support], 0.0)
