# Validation Scope, Context, and Integrity

> *"Different potentials, grid sizes, or experimental noise would still need their own validation passes."*

That sentence is not a hedge; it is the statement of scientific rigor that keeps this project honest. Each simulation we run is a **map** of a particular experimental context, not the territory itself. The moment the context changes, we must redraw the map with new detector events.

## What changes when the context changes?

- **Potentials.** The double-slit notebook uses a free-particle evolution. A harmonic trap, a barrier, or any other potential reshapes the interference pattern. Each new potential must earn agreement by replaying the workflow: evolve \(\psi\), convert to probabilities, sample detector events, and compare with \(\chi^2\) / KS tests.
- **Grid sizes.** Numerical work is an approximation. Coarsening or refining the grid alters dispersion and the precision of \(|\psi|^2\). Robust models should remain stable across reasonable grid choices, and the notebook `grid_resolution_sweep.ipynb` walks through that sensitivity analysis.
- **Experimental noise.** Real detectors click when they should not and miss events they should see. The notebook `noisy_detector_histogram.ipynb` injects a noise model so we can test whether our correlations persist when the conditions mimic the lab.
- **Open-system couplings.** Collapse operators, Lindblad rates, and detector inefficiencies change the statistics even if the Hamiltonian stays fixed. The notebooks `density_matrix_decoherence.ipynb` and `decoherence_cat_state.ipynb` record how quantum trajectories, detector models, and dephasing erase interference and produce classical mixtures.

## Why this strengthens the project

- **Map vs territory.** Acknowledging limits shows that we treat the wave function as a modelling tool. It succeeds only where present-time correlations confirm it.
- **Present-centered validation.** Each notebook is a present-tense measurement: we run the simulation today, under stated assumptions, and record whether statistical tests agree.
- **Built-in physics checks.** Long-run unitarity, energy conservation, and time-step convergence tests keep the numerical core honest before any context-specific data is introduced.
- **Credibility over dogma.** Declaring success everywhere after a single scenario would be unscientific. Documenting the need for fresh validation keeps the project falsifiable and trustworthy.

See the notebooks directory for concrete examples of how the same workflow behaves in different contexts, from ideal interference patterns to decohering qubits with imperfect detectors.
