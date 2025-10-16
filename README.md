# wavefunction-now

Present-centered Schrödinger experiments. This repo demonstrates that the raw empirical content of quantum mechanics is a web of correlations between discrete detector events, and that the wave function \(\psi\) is the mathematical blueprint we use to predict those correlations.

## Key points
- What we measure: binary, localised events (detector clicks) linked by stable statistical patterns in space and time.
- What \(\psi\) does: encodes those patterns via the Born rule (Probability = \(|\psi|^2\)). \(\psi\) itself is not observed; it is a modelling tool for the causal network of events.
- Repo focus: compute \(\psi\), convert it to probabilities, and show—via Monte Carlo sampling—that the same correlation pattern (event histogram) emerges as in the physical apparatus.

## Structure
- `src/wavefunction_now/`: solver and probability utilities.
- `tests/`: pytest suite validating norm conservation, probability sums, and measurement collapse.
- `notebooks/`: exploratory demos (e.g., wave packet spread, double-slit probabilities).

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
