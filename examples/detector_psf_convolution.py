"""Detector point-spread function (PSF) convolution demo.

We start from the present-time probability distribution predicted by the wave
function and apply PSFs of varying widths to model finite detector resolution.
The output shows how the measured distribution depends on the instrument we
deploy now rather than any past narrative about the system.
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


def gaussian_psf(width: float, dx: float, support: float = 5.0) -> np.ndarray:
    """Return a discretised Gaussian PSF."""
    if width <= 0:
        return np.array([1.0])
    half_width = int(math.ceil(support * width / dx))
    offsets = np.arange(-half_width, half_width + 1)
    kernel = np.exp(-(offsets * dx) ** 2 / (2 * width**2))
    kernel /= kernel.sum()
    return kernel


def convolve_distribution(probabilities: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve probabilities with the PSF and renormalise."""
    blurred = np.convolve(probabilities, kernel, mode="same")
    total = blurred.sum()
    if total <= 0:
        raise ValueError("PSF produced zero probability mass")
    return blurred / total


@dataclass
class PSFResult:
    xs: np.ndarray
    ideal: np.ndarray
    measured: dict[float, np.ndarray]


def generate_superposition(sim: SplitStepSimulator) -> np.ndarray:
    psi_left = sim.gaussian_wavepacket(sim.x, x0=-2.0, p0=1.5, sigma=0.6)
    psi_right = sim.gaussian_wavepacket(sim.x, x0=2.0, p0=-1.5, sigma=0.6)
    superposition = psi_left + psi_right
    superposition /= np.linalg.norm(superposition)
    return superposition


def run_psf_demo(widths: list[float]) -> PSFResult:
    sim = SplitStepSimulator(grid_points=1024, length=40.0)
    psi = generate_superposition(sim)
    ideal_prob = born_probability(psi)

    measured: dict[float, np.ndarray] = {}
    for width in widths:
        kernel = gaussian_psf(width, sim.dx)
        measured[width] = convolve_distribution(ideal_prob, kernel)
    return PSFResult(xs=sim.x, ideal=ideal_prob, measured=measured)


def print_summary(result: PSFResult) -> None:
    print("=== Detector PSF impact ===")
    for width, distribution in result.measured.items():
        l1 = np.linalg.norm(distribution - result.ideal, ord=1)
        print(f"PSF width {width:.2f} -> L1 distance {l1:.4f}")


def plot_result(result: PSFResult) -> None:
    if plt is None:
        print("matplotlib not available; skipping plot.")
        return
    plt.figure(figsize=(7, 4))
    plt.plot(result.xs, result.ideal, label="Ideal probability")
    for width, distribution in sorted(result.measured.items()):
        label = "PSF width {:.2f}".format(width)
        plt.plot(result.xs, distribution, label=label)
    plt.xlabel("Position")
    plt.ylabel("Probability density")
    plt.title("Detector PSF convolution of present-time probabilities")
    plt.legend()
    plt.tight_layout()
    plt.savefig("detector_psf_convolution.png", dpi=150)
    plt.close()
    print("Saved plot to detector_psf_convolution.png")


def main() -> None:
    widths = [0.0, 0.2, 0.5]
    result = run_psf_demo(widths)
    print_summary(result)
    plot_result(result)


if __name__ == "__main__":
    main()
