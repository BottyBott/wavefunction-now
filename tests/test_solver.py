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
