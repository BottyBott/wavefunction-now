"""Gaussian wavepacket uncertainty principle demonstration.

We evolve a free Gaussian packet with the split-step simulator and track the
standard deviations Δx and Δp. The product Δx Δp should stay above ħ/2, with the
initial packet saturating the bound. The script prints summary statistics and
optionally plots the time series if matplotlib is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    plt = None

from wavefunction_now.measurement import born_probability
from wavefunction_now.solver import SplitStepSimulator


@dataclass
class UncertaintyRecord:
    times: np.ndarray
    delta_x: np.ndarray
    delta_p: np.ndarray
    product: np.ndarray


def run_demo() -> UncertaintyRecord:
    sim = SplitStepSimulator(grid_points=512, length=30.0)
    x0 = -5.0
    p0 = 2.0
    sigma = 1.0
    psi = sim.gaussian_wavepacket(sim.x, x0=x0, p0=p0, sigma=sigma)
    potential = np.zeros_like(sim.x)

    dt = 0.01
    steps = 400

    times = np.arange(steps, dtype=float) * dt
    delta_x = np.zeros(steps)
    delta_p = np.zeros(steps)

    state = psi
    for idx in range(steps):
        prob_x = born_probability(state)
        mean_x = float(np.dot(prob_x, sim.x))
        mean_x_sq = float(np.dot(prob_x, sim.x**2))
        var_x = mean_x_sq - mean_x**2

        psi_fft = np.fft.fft(state)
        derivative = np.fft.ifft(1j * sim.k * psi_fft)
        second_derivative = np.fft.ifft(-(sim.k**2) * psi_fft)
        p_expectation = np.real(-1j * sim.hbar * np.vdot(state, derivative) * sim.dx)
        p2_expectation = np.real(-sim.hbar**2 * np.vdot(state, second_derivative) * sim.dx)
        var_p = max(p2_expectation - p_expectation**2, 0.0)

        delta_x[idx] = math.sqrt(max(var_x, 0.0))
        delta_p[idx] = math.sqrt(var_p)

        state = sim.evolve(state, potential, dt=dt, steps=1)

    product = delta_x * delta_p
    return UncertaintyRecord(times=times, delta_x=delta_x, delta_p=delta_p, product=product)


def describe(record: UncertaintyRecord, hbar: float) -> None:
    print("=== Uncertainty summary ===")
    print(f"Initial Δx: {record.delta_x[0]:.6f}")
    print(f"Initial Δp: {record.delta_p[0]:.6f}")
    print(f"Initial product: {record.product[0]:.6f}")
    print(f"Minimum product: {record.product.min():.6f}")
    print(f"Hbar/2: {0.5 * hbar:.6f}")


def plot(record: UncertaintyRecord, hbar: float) -> None:
    if plt is None:
        print("matplotlib not available; skipping plot.")
        return
    plt.figure(figsize=(7, 4))
    plt.plot(record.times, record.delta_x, label=r"$\Delta x$")
    plt.plot(record.times, record.delta_p, label=r"$\Delta p$")
    plt.plot(record.times, record.product, label=r"$\Delta x \Delta p$")
    plt.axhline(0.5 * hbar, color="black", linestyle="--", label=r"$\hbar/2$")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Gaussian wavepacket uncertainty evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("uncertainty_gaussian.png", dpi=150)
    plt.close()
    print("Saved plot to uncertainty_gaussian.png")


def main() -> None:
    record = run_demo()
    describe(record, hbar=1.0)
    plot(record, hbar=1.0)


if __name__ == "__main__":
    main()
