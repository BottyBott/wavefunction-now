# Present-Centered Programme

This project is a narrow, computational slice of a broader reality framework. It is based on my book-length thesis that contests the habit of treating models as time machines. It insists that science earns authority only when it remains anchored to the events we can register **now**. The wavefunction-now repository operationalises that stance with code.

## What we are doing

- **Measure events, not myths.** Detectors click. Those clicks arrive as localised, binary events. Everything we claim about a quantum experiment must be traceable to the correlations among those events.
- **Use \(\psi\) as a blueprint, not a being.** The wave function is a calculational tool that converts preparations into probability distributions. It is not a substance, a past history, or a hidden ontology. We treat it as a map drawn to match the frequencies we witness in the lab.
- **Stay in the present tense.** Every simulation is resolved in real time. We evolve \(\psi\) under explicit constraints, generate Monte Carlo events, and ask whether today’s statistical tests agree. No inference about the past or future is accepted without new measurements.
- **Model decoherence without time-travel.** Density matrices, Lindblad generators, and quantum trajectories let us include environments and detector imperfections while keeping the focus on today’s detector statistics.

## Why we are doing it

- **Reality is systemic and present.** The thesis argues that wholes—not particles or origin stories—carry causal authority. Systems earn their laws by sustaining coherent behaviour in the moment. By putting the entire workflow on present-time footing, we demonstrate how a systemic account can coexist with standard quantum numerics.
- **Validation replaces extrapolation.** Classical narratives rely on extrapolating empirical laws backward to an assumed origin. We replace that move with explicit validation loops: compute |ψ|², sample detector events, compare histograms (χ², KS), and repeat whenever the context changes.
- **Demonstrate rigor without teleology.** The repository is a living proof-of-principle. It shows that one can simulate interference, harmonic confinement, coarse grids, and noisy detectors while acknowledging that each success is context-bound. Agreement is earned scenario by scenario, not granted by myth.

## How the code ties in

1. **Split-step simulator (`src/solver.py`).** Generates present-time wave functions for specified potentials without asserting anything about their origins.
2. **Measurement utilities (`src/measurement.py`).** Convert wave functions and density matrices into probability histograms and statistical tests, making the link from model to detector explicit.
3. **Open-system solvers (`src/lindblad.py`).** Provide Lindblad master equations and quantum jump trajectories so decoherence and detector inefficiency are simulated alongside pure-state dynamics.
4. **Notebooks (`notebooks/*.ipynb`).** Narrate different contexts—double-slit, harmonic oscillator, grid sweeps, noisy or lossy detectors—and document the validation outcome for each.
5. **Examples (`examples/*.py`).** Quick, executable sanity checks—e.g. Ehrenfest expectation dynamics—that keep the focus on present-time constraints without invoking hidden trajectories.
6. **Tests (`tests/*.py`).** Automate checks that maintain internal coherence (normalisation, long-run unitarity, energy conservation, measurement collapse, time-step convergence) while leaving room for new, scenario-specific assertions.

This project gives us a sandbox where the philosophy of present-centred science can be inspected, modified, and extended in executable form as it relates to the quantum wave function. Every new notebook is an experiment in the systemic sense—an encounter with a whole under specified constraints, judged by what its events deliver right now.
