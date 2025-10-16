"""Time-dependent Schrödinger solver using split-step Fourier method.

The goal is to compute the predictive kernel that links preparation events to
detector statistics. The wave function \(\psi\) is not treated as a physical
entity, but as a compact description of the correlations between discrete
measurement outcomes. Everything we return can be mapped to probability
distributions (\(|\psi|^2\)) or expectation values that summarise those
correlations.
"""

from __future__ import annotations

import numpy as np


class SplitStepSimulator:
    """Evolve a 1-D wave packet under a potential using split-step method."""

    def __init__(self, grid_points: int, length: float, mass: float = 1.0, hbar: float = 1.0):
        self.grid_points = grid_points
        self.length = length
        self.mass = mass
        self.hbar = hbar
        self.dx = length / grid_points
        self.x = np.linspace(-length / 2, length / 2, grid_points, endpoint=False)
        self.k = 2 * np.pi * np.fft.fftfreq(grid_points, d=self.dx)

    def evolve(self, psi0: np.ndarray, potential: np.ndarray, dt: float, steps: int) -> np.ndarray:
        """Evolve `psi0` forward in time by `steps` increments of size `dt`."""
        psi = psi0.copy()
        kinetic_phase = np.exp(-1j * (self.hbar * self.k ** 2 / (2 * self.mass)) * dt / self.hbar)
        for _ in range(steps):
            psi *= np.exp(-1j * potential * dt / (2 * self.hbar))
            psi_k = np.fft.fft(psi)
            psi_k *= kinetic_phase
            psi = np.fft.ifft(psi_k)
            psi *= np.exp(-1j * potential * dt / (2 * self.hbar))
        return psi

    @staticmethod
    def gaussian_wavepacket(x: np.ndarray, x0: float, p0: float, sigma: float) -> np.ndarray:
        """Return a normalised Gaussian wave packet."""
        norm = (1 / (sigma * np.sqrt(np.pi))) ** 0.5
        return norm * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * p0 * x)
