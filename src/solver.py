r"""Time-dependent Schrödinger solver using split-step Fourier method.

The goal is to compute the predictive kernel that links preparation events to
detector statistics. The wave function \(\psi\) is not treated as a physical
entity, but as a compact description of the correlations between discrete
measurement outcomes. Everything we return can be mapped to probability
distributions (\(|\psi|^2\)) or expectation values that summarise those
correlations.
"""

from __future__ import annotations

import numpy as np

from .measurement import projective_measurement


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
        if psi0.shape != (self.grid_points,):
            msg = f"psi0 shape {psi0.shape} does not match simulator grid ({self.grid_points},)"
            raise ValueError(msg)
        if potential.shape != psi0.shape:
            msg = "potential must have the same shape as psi0"
            raise ValueError(msg)
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if steps == 0:
            return psi0.copy()
        if dt <= 0:
            raise ValueError("time step dt must be positive")

        psi = psi0.copy()
        # Phase factor associated with the kinetic evolution in momentum space.
        kinetic_phase = np.exp(-1j * (self.hbar * self.k ** 2 / (2 * self.mass)) * dt)
        # Half-step potential phase used before and after the kinetic update.
        potential_phase = np.exp(-1j * potential * dt / (2 * self.hbar))
        for _ in range(steps):
            psi *= potential_phase
            psi_k = np.fft.fft(psi)
            psi_k *= kinetic_phase
            psi = np.fft.ifft(psi_k)
            psi *= potential_phase
        return psi

    def position_projectors(self, bin_edges: np.ndarray) -> np.ndarray:
        """Return diagonal projectors that coarse-grain position into bins."""
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bin_edges must be a one-dimensional array with at least two entries")
        if not np.all(np.isfinite(edges)):
            raise ValueError("bin_edges must be finite")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("bin_edges must be strictly increasing")

        masks: list[np.ndarray] = []
        for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            if idx == edges.size - 2:
                mask = (self.x >= left) & (self.x <= right)
            else:
                mask = (self.x >= left) & (self.x < right)
            if not mask.any():
                raise ValueError("each bin must contain at least one grid point")
            masks.append(mask.astype(float))
        return np.stack(masks, axis=0)

    def bin_probabilities(self, psi: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
        """Aggregate |psi|^2 into coarse spatial bins."""
        psi = np.asarray(psi, dtype=complex)
        if psi.shape != (self.grid_points,):
            msg = f"psi shape {psi.shape} does not match simulator grid ({self.grid_points},)"
            raise ValueError(msg)
        projectors = self.position_projectors(bin_edges)
        probabilities = np.sum(np.abs(projectors * psi) ** 2, axis=1)
        total = probabilities.sum()
        if total <= 0:
            raise ValueError("wave function has zero norm")
        return probabilities / total

    def measure_position(
        self,
        psi: np.ndarray,
        bin_edges: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Perform a coarse-grained position measurement with projective collapse."""
        psi = np.asarray(psi, dtype=complex)
        if psi.shape != (self.grid_points,):
            msg = f"psi shape {psi.shape} does not match simulator grid ({self.grid_points},)"
            raise ValueError(msg)
        projectors = self.position_projectors(bin_edges)
        outcome, post_state, probabilities = projective_measurement(psi, projectors, rng)
        return outcome, post_state, probabilities

    @staticmethod
    def gaussian_wavepacket(x: np.ndarray, x0: float, p0: float, sigma: float) -> np.ndarray:
        """Return a normalised Gaussian wave packet."""
        norm = (1 / (sigma * np.sqrt(np.pi))) ** 0.5
        return norm * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * p0 * x)
