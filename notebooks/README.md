# Notebooks Roadmap

These notebooks show how discrete detector events build up empirical
correlations that match \(|\psi|^2\) predictions across different contexts.

## Current notebooks

1. `double_slit_histogram.ipynb`
   - Free-particle propagation of a two-slit superposition.
   - Accumulates detector hits and compares histograms via \(\chi^2\)/KS tests.

2. `harmonic_oscillator_histogram.ipynb`
   - Bounded potential with stationary-state sampling.
   - Confirms that detector events match the harmonic ground-state profile.

3. `grid_resolution_sweep.ipynb`
   - Repeats the double-slit scenario at multiple grid resolutions.
   - Shows how p-values behave as the discretisation is coarsened or refined.

4. `noisy_detector_histogram.ipynb`
   - Injects Poisson noise into detector counts.
   - Illustrates how agreement degrades and how the tests quantify the shift.

5. `free_wavepacket.ipynb`
   - Tracks a single Gaussian packet in free space.
   - Validates the spreading wave against detector statistics.

6. `stern_gerlach_correlations.ipynb`
   - Simulates sequential Stern–Gerlach measurements.
   - Demonstrates collapse-induced correlations for aligned and orthogonal axes.

7. `density_matrix_decoherence.ipynb`
   - Evolves an amplitude-damped qubit with Lindblad master equations and quantum trajectories.
   - Mixes in detector inefficiency and dark counts to show how density matrices drive realistic click statistics.

8. `decoherence_cat_state.ipynb`
   - Starts from a macroscopic superposition and applies dephasing Lindblad dynamics.
   - Shows off-diagonal decay, σx expectation collapse, and trajectory averaging that converges to a classical mixture.

Each notebook emphasises that detectors record binary events and the wave
function is the blueprint that predicts their correlation structure.
