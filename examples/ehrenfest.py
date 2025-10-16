"""Ehrenfest theorem sanity checks for the split-step simulator.

This script keeps everything in the present tense: it evolves a state with the
existing solver, records expectation values ⟨x⟩ and ⟨p⟩, and verifies that they
obey the classical-looking differential relations implied by the Schrödinger
equation. No hidden trajectories are invoked—just constraints on observable
expectations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wavefunction_now.measurement import born_probability
from wavefunction_now.solver import SplitStepSimulator


def expectation_x(sim: SplitStepSimulator, psi: np.ndarray) -> float:
    return float(np.dot(born_probability(psi), sim.x))


def expectation_p(sim: SplitStepSimulator, psi: np.ndarray) -> float:
    """Return ⟨p⟩ using spectral differentiation."""
    psi_k = np.fft.fft(psi)
    derivative = np.fft.ifft(1j * sim.k * psi_k)
    expectation = -1j * sim.hbar * np.vdot(psi, derivative) * sim.dx
    return float(np.real(expectation))


def expectation_force(probabilities: np.ndarray, grad_potential: np.ndarray) -> float:
    """Return ⟨∂V/∂x⟩ given pointwise probabilities and gradient."""
    return float(np.dot(probabilities, grad_potential))


@dataclass
class EhrenfestResult:
    times: np.ndarray
    position_expectation: np.ndarray
    momentum_expectation: np.ndarray
    velocity_expectation: np.ndarray
    force_expectation: np.ndarray
    velocity_residual: float
    force_residual: float
    mass: float
    omega: float | None = None


def run_ehrenfest_check(
    sim: SplitStepSimulator,
    psi0: np.ndarray,
    potential: np.ndarray,
    dt: float,
    steps: int,
) -> EhrenfestResult:
    """Evolve psi and return expectation-value diagnostics."""

    grad_potential = np.gradient(potential, sim.dx, edge_order=2)

    psi = psi0.copy()
    times = np.arange(steps, dtype=float) * dt
    xs: list[float] = []
    ps: list[float] = []
    forces: list[float] = []

    for _ in range(steps):
        prob = born_probability(psi)
        xs.append(float(np.dot(prob, sim.x)))
        ps.append(expectation_p(sim, psi))
        forces.append(expectation_force(prob, grad_potential))
        psi = sim.evolve(psi, potential, dt=dt, steps=1)

    xs_arr = np.asarray(xs)
    ps_arr = np.asarray(ps)
    forces_arr = np.asarray(forces)

    dx_dt = np.gradient(xs_arr, dt)
    dp_dt = np.gradient(ps_arr, dt)

    velocity_rhs = ps_arr / sim.mass
    velocity_residual = math.sqrt(np.mean((dx_dt - velocity_rhs) ** 2))
    force_residual = math.sqrt(np.mean((dp_dt + forces_arr) ** 2))

    return EhrenfestResult(
        times=times,
        position_expectation=xs_arr,
        momentum_expectation=ps_arr,
        velocity_expectation=dx_dt,
        force_expectation=-forces_arr,
        velocity_residual=velocity_residual,
        force_residual=force_residual,
        mass=sim.mass,
    )


def demo_free_particle() -> EhrenfestResult:
    sim = SplitStepSimulator(grid_points=512, length=30.0)
    psi0 = sim.gaussian_wavepacket(sim.x, x0=-6.0, p0=2.0, sigma=1.0)
    potential = np.zeros_like(sim.x)
    return run_ehrenfest_check(sim, psi0, potential, dt=0.01, steps=400)


def demo_harmonic() -> EhrenfestResult:
    sim = SplitStepSimulator(grid_points=512, length=30.0)
    omega = 0.4
    potential = 0.5 * sim.mass * omega**2 * sim.x**2
    sigma = math.sqrt(sim.hbar / (sim.mass * omega))
    psi0 = sim.gaussian_wavepacket(sim.x, x0=2.0, p0=0.0, sigma=sigma)
    result = run_ehrenfest_check(sim, psi0, potential, dt=0.01, steps=800)
    result.omega = omega
    return result


def print_summary(name: str, result: EhrenfestResult, expected_force: np.ndarray) -> None:
    print(f"=== {name} ===")
    print(f"Velocity residual rms   : {result.velocity_residual:.3e}")
    print(f"Force residual rms      : {result.force_residual:.3e}")
    if expected_force is not None:
        wrapped_error = math.sqrt(np.mean((result.force_expectation - expected_force) ** 2))
        print(f"Analytical force RMS err: {wrapped_error:.3e}")
    print()


def main() -> None:
    free = demo_free_particle()
    print_summary("Free particle", free, expected_force=np.zeros_like(free.force_expectation))

    harmonic = demo_harmonic()
    omega = getattr(harmonic, "omega", 0.0)
    analytical_force = -harmonic.mass * omega**2 * harmonic.position_expectation
    print_summary("Harmonic oscillator", harmonic, expected_force=analytical_force)


if __name__ == "__main__":
    main()
