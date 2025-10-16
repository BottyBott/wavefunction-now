import numpy as np
import pytest

from wavefunction_now.solver import SplitStepSimulator


def kinetic_energy(sim: SplitStepSimulator, psi: np.ndarray) -> float:
    """Return ⟨T⟩ for a state in the simulator grid."""
    psi_k = np.fft.fft(psi)
    energies = (sim.hbar ** 2 * (sim.k ** 2)) / (2 * sim.mass)
    density = np.abs(psi_k) ** 2 / psi.size
    return float(np.sum(energies * density))


def potential_energy(potential: np.ndarray, psi: np.ndarray, dx: float) -> float:
    """Return ⟨V⟩ given a potential evaluated on the grid."""
    expectation = np.vdot(psi, potential * psi)
    return float(np.real(expectation) * dx)


@pytest.mark.parametrize(
    ("grid_points", "length", "x0", "p0", "sigma"),
    [
        (512, 20.0, -2.0, 2.5, 0.7),
        (512, 20.0, 0.0, 0.0, 1.0),
    ],
)
def test_energy_conservation_free_particle(grid_points, length, x0, p0, sigma):
    """Free particle: ⟨H⟩ should remain constant under evolution."""
    sim = SplitStepSimulator(grid_points=grid_points, length=length)
    psi = sim.gaussian_wavepacket(sim.x, x0=x0, p0=p0, sigma=sigma)
    potential = np.zeros_like(sim.x)

    dt = 0.01
    steps = 200

    energies = []
    current = psi
    for _ in range(steps):
        k_energy = kinetic_energy(sim, current)
        v_energy = potential_energy(potential, current, sim.dx)
        energies.append(k_energy + v_energy)
        current = sim.evolve(current, potential, dt=dt, steps=1)

    energies = np.array(energies)
    assert np.allclose(energies, energies[0], atol=1e-3)
