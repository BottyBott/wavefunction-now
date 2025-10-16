r"""Density-matrix evolutions: Lindblad master equation and quantum trajectories.

These tools complement the split-step wave-function solver by providing an
open-system description. The master equation evolves density matrices under
decoherence, while the trajectory simulator samples individual quantum jump
realizations that average to the same dynamics. Both enable modelling realistic
detectors whose imperfections require density matrices instead of pure states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


def _as_square_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a complex square matrix."""
    arr = np.asarray(matrix, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return arr


def _project_to_physical(rho: np.ndarray) -> np.ndarray:
    """Numerically enforce Hermiticity and unit trace."""
    hermitian = 0.5 * (rho + rho.conj().T)
    trace = np.trace(hermitian)
    if np.isclose(trace, 0.0):
        raise ValueError("density matrix trace vanished")
    return hermitian / trace


@dataclass(slots=True)
class LindbladSimulator:
    """Deterministic density-matrix evolution under the Lindblad equation."""

    hamiltonian: np.ndarray
    collapse_operators: tuple[np.ndarray, ...] = field(default_factory=tuple)
    hbar: float = 1.0
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        self.hamiltonian = _as_square_matrix(self.hamiltonian, "hamiltonian")
        self.dimension = self.hamiltonian.shape[0]
        collapse_ops: list[np.ndarray] = []
        for idx, op in enumerate(self.collapse_operators):
            arr = _as_square_matrix(op, f"collapse_operators[{idx}]")
            if arr.shape != self.hamiltonian.shape:
                raise ValueError("collapse operators must match the Hamiltonian dimension")
            collapse_ops.append(arr)
        self.collapse_operators = tuple(collapse_ops)
        if self.hbar <= 0:
            raise ValueError("hbar must be positive")

    def _rhs(self, rho: np.ndarray) -> np.ndarray:
        """Time derivative of the density matrix."""
        commutator = self.hamiltonian @ rho - rho @ self.hamiltonian
        derivative = (-1j / self.hbar) * commutator
        for c_op in self.collapse_operators:
            c_dag = c_op.conj().T
            jump_term = c_op @ rho @ c_dag
            anticommutator = c_dag @ c_op @ rho + rho @ c_dag @ c_op
            derivative += jump_term - 0.5 * anticommutator
        return derivative

    def evolve(self, rho0: np.ndarray, times: Iterable[float]) -> np.ndarray:
        """Return density matrices evaluated at the requested `times`.

        Parameters
        ----------
        rho0:
            Initial density matrix (Hermitian, unit trace).
        times:
            Iterable of monotonically increasing time stamps. The first entry is
            interpreted as the initial time and should typically be zero.
        """

        rho = _as_square_matrix(rho0, "rho0")
        if rho.shape != self.hamiltonian.shape:
            raise ValueError("rho0 dimension does not match Hamiltonian")
        rho = _project_to_physical(rho)

        times_arr = np.asarray(list(times), dtype=float)
        if times_arr.ndim != 1 or times_arr.size < 2:
            raise ValueError("times must contain at least two entries")
        if not np.all(np.diff(times_arr) > 0):
            raise ValueError("times must be strictly increasing")

        results = np.empty((times_arr.size, self.dimension, self.dimension), dtype=complex)
        results[0] = rho
        current = rho
        for idx in range(1, times_arr.size):
            dt = times_arr[idx] - times_arr[idx - 1]
            current = self._rk4_step(current, dt)
            results[idx] = current
        return results

    def _rk4_step(self, rho: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0:
            raise ValueError("time step must be positive")
        k1 = self._rhs(rho)
        k2 = self._rhs(rho + 0.5 * dt * k1)
        k3 = self._rhs(rho + 0.5 * dt * k2)
        k4 = self._rhs(rho + dt * k3)
        updated = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return _project_to_physical(updated)


@dataclass(slots=True)
class QuantumTrajectorySimulator:
    """Monte Carlo wave-function (quantum jump) solver."""

    hamiltonian: np.ndarray
    collapse_operators: tuple[np.ndarray, ...]
    hbar: float = 1.0
    dimension: int = field(init=False)
    _collapse_rates: tuple[np.ndarray, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.hamiltonian = _as_square_matrix(self.hamiltonian, "hamiltonian")
        self.dimension = self.hamiltonian.shape[0]
        collapse_ops: list[np.ndarray] = []
        for idx, op in enumerate(self.collapse_operators):
            arr = _as_square_matrix(op, f"collapse_operators[{idx}]")
            if arr.shape != self.hamiltonian.shape:
                raise ValueError("collapse operators must match the Hamiltonian dimension")
            collapse_ops.append(arr)
        if not collapse_ops:
            raise ValueError("quantum trajectories require at least one collapse operator")
        self.collapse_operators = tuple(collapse_ops)
        if self.hbar <= 0:
            raise ValueError("hbar must be positive")
        self._collapse_rates = tuple(op.conj().T @ op for op in self.collapse_operators)

    def trajectory(
        self,
        psi0: np.ndarray,
        times: Iterable[float],
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate a single quantum trajectory."""

        psi = np.asarray(psi0, dtype=complex)
        if psi.ndim != 1 or psi.size != self.dimension:
            raise ValueError("psi0 dimension must match Hamiltonian")
        norm = np.linalg.norm(psi)
        if np.isclose(norm, 0.0):
            raise ValueError("psi0 cannot be the zero vector")
        psi = psi / norm

        if rng is None:
            rng = np.random.default_rng()

        times_arr = np.asarray(list(times), dtype=float)
        if times_arr.ndim != 1 or times_arr.size < 2:
            raise ValueError("times must contain at least two entries")
        if not np.all(np.diff(times_arr) > 0):
            raise ValueError("times must be strictly increasing")

        evolution = np.empty((times_arr.size, self.dimension), dtype=complex)
        evolution[0] = psi
        current = psi
        for idx in range(1, times_arr.size):
            dt = times_arr[idx] - times_arr[idx - 1]
            current = self._step(current, dt, rng)
            evolution[idx] = current
        return evolution

    def ensemble_density_matrix(
        self,
        psi0: np.ndarray,
        times: Iterable[float],
        trajectories: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Average density matrix inferred from repeated trajectories."""
        if trajectories <= 0:
            raise ValueError("trajectories must be positive")
        if rng is None:
            rng = np.random.default_rng()

        times_arr = np.asarray(list(times), dtype=float)
        density = np.zeros((times_arr.size, self.dimension, self.dimension), dtype=complex)
        for _ in range(trajectories):
            traj_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            psi_evolution = self.trajectory(psi0, times_arr, rng=traj_rng)
            density += np.einsum("ti,tj->tij", psi_evolution, psi_evolution.conj())
        density /= trajectories
        for idx in range(density.shape[0]):
            density[idx] = _project_to_physical(density[idx])
        return density

    def _step(self, psi: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
        """Perform one stochastic step using the first-order quantum jump update."""
        if dt <= 0:
            raise ValueError("time step must be positive")

        h_eff = self.hamiltonian.copy()
        for rate in self._collapse_rates:
            h_eff = h_eff - 0.5j * self.hbar * rate

        # Non-unitary evolution
        evolution_op = np.eye(self.dimension, dtype=complex) - (1j * dt / self.hbar) * h_eff
        tentative = evolution_op @ psi
        norm_sq = float(np.real(np.vdot(tentative, tentative)))

        if norm_sq < 0.0:
            raise ValueError("non-physical norm encountered during trajectory step")

        jump_probabilities = np.array(
            [dt * float(np.real(np.vdot(psi, rate @ psi))) for rate in self._collapse_rates],
            dtype=float,
        )
        jump_total = jump_probabilities.sum()

        if jump_total < 0:
            raise ValueError("negative jump probability encountered")

        if jump_total > 1.0:
            raise ValueError("time step too large for stable quantum trajectory simulation")

        zeta = rng.uniform()
        if zeta < jump_total:
            if jump_total == 0:
                raise ValueError("jump probability requested but total probability is zero")
            weights = jump_probabilities / jump_total
            channel = rng.choice(len(self.collapse_operators), p=weights)
            new_state = self.collapse_operators[channel] @ psi
            norm = np.linalg.norm(new_state)
            if np.isclose(norm, 0.0):
                raise ValueError("collapse operator annihilated the state during trajectory")
            return new_state / norm

        if np.isclose(norm_sq, 0.0):
            raise ValueError("state norm vanished without a jump")
        return tentative / np.sqrt(norm_sq)
