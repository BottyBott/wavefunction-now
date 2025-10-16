import numpy as np
import pytest

from wavefunction_now.solver import SplitStepSimulator
from wavefunction_now.measurement import born_probability


@pytest.mark.parametrize(
    ("grid_points", "length", "dt", "steps"),
    [
        (512, 20.0, 0.01, 500),
        (512, 20.0, 0.005, 1000),
    ],
)
def test_long_evolution_preserves_norm(grid_points, length, dt, steps):
    """Ensure the split-step propagator remains unitary over long runs."""
    sim = SplitStepSimulator(grid_points=grid_points, length=length)
    psi0 = sim.gaussian_wavepacket(sim.x, x0=-1.5, p0=2.5, sigma=0.7)
    potential = np.zeros_like(sim.x)

    psi_t = sim.evolve(psi0, potential, dt=dt, steps=steps)
    norm_initial = born_probability(psi0).sum()
    norm_final = born_probability(psi_t).sum()

    assert pytest.approx(norm_initial, rel=1e-6, abs=1e-6) == norm_final


def test_time_step_convergence_improves_with_smaller_dt():
    """Verify that reducing dt produces solutions closer to a reference."""
    sim = SplitStepSimulator(grid_points=512, length=20.0)
    psi0 = sim.gaussian_wavepacket(sim.x, x0=-1.0, p0=1.8, sigma=0.6)
    potential = np.zeros_like(sim.x)
    total_time = 0.5

    reference_dt = 0.0025
    reference_steps = int(total_time / reference_dt)
    psi_reference = sim.evolve(psi0, potential, dt=reference_dt, steps=reference_steps)

    trial_dts = [0.02, 0.01, 0.005]
    errors = []
    for dt in trial_dts:
        steps = int(total_time / dt)
        psi_trial = sim.evolve(psi0, potential, dt=dt, steps=steps)
        error = np.linalg.norm(psi_trial - psi_reference)
        errors.append(error)

    # Ensure errors shrink as dt shrinks; allow small numerical slack.
    assert errors[0] > errors[1] * 1.05
    assert errors[1] > errors[2] * 1.05
