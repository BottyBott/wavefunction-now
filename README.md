# wavefunction-now

Present-centered Schrödinger experiments. This repo demonstrates that the raw empirical content of quantum mechanics is a web of correlations between discrete detector events, and that the wave function \(\psi\) is the mathematical blueprint we use to predict those correlations.

## Key points
- What we measure: binary, localised events (detector clicks) linked by stable statistical patterns in space and time.
- Where we stand: reality is encountered in the present; models earn trust only by matching the events we can register now.
- What \(\psi\) does: encodes those patterns via the Born rule (Probability = \(|\psi|^2\)). \(\psi\) itself is not observed; it is a modelling tool for the causal network of events.
- Repo focus: compute \(\psi\), convert it to probabilities, and show—via Monte Carlo sampling—that the same correlation pattern (event histogram) emerges as in the physical apparatus.
- Open systems: evolve density matrices with Lindblad generators or quantum trajectories so that decoherence and realistic detector models live in the same workflow.

## Structure
- `src/`: installable `wavefunction_now` package with split-step solvers, Lindblad/trajectory engines, and measurement + detector utilities.
- `tests/`: pytest suite validating long-run unitarity, energy conservation, probability sums, measurement collapse, and time-step convergence.
- `examples/`: standalone scripts highlighting expectation-value constraints (Ehrenfest), etc.
- `notebooks/`: exploratory demos (double-slit, harmonic trap, grid sweeps, noisy detectors, decoherence trajectories).
- `docs/`: conceptual explanations and validation notes.

## Getting started
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Roadmap
- Implement split-step Fourier time evolution (the predictive kernel).
- Add measurement operators and histogram comparisons showing event patterns.
- Document how the simulated correlations map to actual detector statistics (double-slit, Stern–Gerlach, etc.).

## Falsifiability
This repo stays neutral. The workflow is:
1. Compute \(\psi\) for a preparation and convert it to predicted probabilities.
2. Generate Monte Carlo samples that represent detector events.
3. Compare predicted and sampled distributions (KS/chi-squared tests in notebooks, unit tests for conservation laws).

If the comparisons fail beyond statistical tolerance, we treat the model (or its numerical implementation) as falsified and record the discrepancy rather than forcing agreement. Agreement must earn its way through measurement; disagreement is an acceptable—and documented—outcome.

## Context-bound validation (map vs territory)
- Passing the double-slit notebook confirms that, for a free-particle propagation with a 2048-point grid and ideal detectors, the sampled events reproduce \(|\psi|^2\) according to both \(\chi^2\) and KS statistics.
- Changing the context—e.g. introducing new potentials, coarser/finer grids, or detector noise—requires re-running the validation workflow. Each scenario is its own "micro-law" that must earn agreement in the present tense.
- Open systems—e.g. the amplitude-damped qubit in the density-matrix notebook—need their own validation loop because collapse operators, detector inefficiencies, and Monte Carlo trajectories introduce fresh failure modes.
- This is not a weakness; it embodies the project's philosophy that the wave function is a modelling tool whose adequacy is continually tested against the correlations we observe.

See `docs/present_centered_program.md` for the conceptual motivation and `docs/validation_scope.md` plus the notebooks listed below for scenario-specific checks.

### Notebooks at a glance
- `double_slit_histogram.ipynb`: interference verification (free particle).
- `harmonic_oscillator_histogram.ipynb`: bounded potential and stationary-state sampling.
- `grid_resolution_sweep.ipynb`: sensitivity of predictions to grid coarsening/refinement.
- `noisy_detector_histogram.ipynb`: robustness of correlations under detector noise.
- `free_wavepacket.ipynb`: Gaussian packet propagation and sampling.
- `stern_gerlach_correlations.ipynb`: sequential spin measurement correlations.
- `density_matrix_decoherence.ipynb`: Lindblad evolution, quantum trajectories, and detector imperfections for an amplitude-damped qubit.

Run the notebooks to explore how the same workflow behaves as the physical assumptions shift.
