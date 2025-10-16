import numpy as np
import pytest

from wavefunction_now.lindblad import LindbladSimulator, QuantumTrajectorySimulator
from wavefunction_now.measurement import (
    apply_detector_response,
    density_matrix_probabilities,
)
from wavefunction_now.solver import SplitStepSimulator


def test_lindblad_amplitude_damping_trace_and_population():
    gamma = 0.4
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    collapse = (np.sqrt(gamma) * sigma_minus,)
    hamiltonian = np.zeros((2, 2), dtype=complex)
    simulator = LindbladSimulator(hamiltonian, collapse)

    rho0 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    times = np.linspace(0.0, 4.0, 81)
    evolution = simulator.evolve(rho0, times)

    assert evolution.shape == (times.size, 2, 2)
    final_rho = evolution[-1]
    assert np.isclose(np.trace(final_rho), 1.0, atol=1e-6)

    expected_excited = np.exp(-gamma * times[-1])
    assert pytest.approx(np.real(final_rho[1, 1]), rel=0.1) == expected_excited


def test_quantum_trajectory_matches_master_equation_mean():
    gamma = 0.3
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    collapse = (np.sqrt(gamma) * sigma_minus,)
    hamiltonian = np.zeros((2, 2), dtype=complex)

    lindblad = LindbladSimulator(hamiltonian, collapse)
    trajectories = QuantumTrajectorySimulator(hamiltonian, collapse)

    rho0 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    psi0 = np.array([0.0, 1.0], dtype=complex)

    times = np.linspace(0.0, 0.8, 81)
    master_evolution = lindblad.evolve(rho0, times)
    rng = np.random.default_rng(42)
    ensemble = trajectories.ensemble_density_matrix(psi0, times, trajectories=300, rng=rng)

    master_excited = np.real(master_evolution[-1, 1, 1])
    ensemble_excited = np.real(ensemble[-1, 1, 1])
    assert pytest.approx(ensemble_excited, rel=0.2, abs=0.05) == master_excited


def test_density_matrix_probabilities_match_projective_masks():
    sim = SplitStepSimulator(grid_points=64, length=10.0)
    psi = sim.gaussian_wavepacket(sim.x, x0=-1.0, p0=0.4, sigma=0.7)
    rho = np.outer(psi, psi.conj())
    bins = np.array([-5.0, 0.0, 5.0])
    projectors = sim.position_projectors(bins)

    pure_probs = sim.bin_probabilities(psi, bins)
    rho_probs = density_matrix_probabilities(rho, projectors)

    assert pytest.approx(rho_probs.sum(), rel=1e-6) == 1.0
    assert np.allclose(rho_probs, pure_probs, atol=1e-6)


def test_apply_detector_response_mixes_background():
    probabilities = np.array([0.7, 0.3])
    adjusted = apply_detector_response(
        probabilities,
        efficiency=0.8,
        dark_count=0.2,
        background=np.array([0.25, 0.75]),
    )
    assert pytest.approx(adjusted.sum(), rel=1e-12) == 1.0
    assert np.allclose(adjusted, np.array([0.61, 0.39]), atol=1e-8)
