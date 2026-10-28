# isingzeros

Locating partition-function zeros of the Ising model with return probabilities of kicked-Ising circuits.

This package contains the circuits, exact references, emulation, analysis and resource planning for **ISINGZEROS**, a project for the EuroHPC JU Quantum Access Pilot (Euro-Q-Exa at LRZ, with an optional cross-check on PIAST-Q at PCSS).

## The idea in brief

Take a two-dimensional Ising strip: rows of L sites closed into a ring, M rows, and free first and last rows. At imaginary couplings, its partition function is the amplitude for a kicked-Ising circuit to return to its initial state:

    Z_strip ∝ <+^L| V_y (V_x V_y)^(M-1) |+^L>,   V_y = ∏_k exp(iθ Z_k Z_k+1),   V_x = ∏_k exp(iφ X_k).

Zeros of the partition function therefore appear as circuit angles where the return probability P(θ) = |<+|U|+>|² vanishes. Most zeros lie off this "unitary axis".

P(θ) is a trigonometric polynomial in θ. By the Fejér–Riesz theorem, a fit to P(θ) measured on the axis gives the nearby off-axis zeros up to a mirror image, and a short post-selected two-point test decides between the two. The package also implements the established route, direct post-selection with a linear combination of unitaries (LCU), so the two can be compared under the same noise.

Every circuit in the project has an exact classical reference from a row transfer matrix. Every measured number can therefore be checked.

## Contents

| Module | What it does |
|:--|:--|
| `classical.py` | Exact partition functions at complex coupling: brute force for testing, and a row transfer matrix with dense vectors up to 24 qubits; zero search |
| `circuits.py` | Qiskit circuits: return probability on rings and layered slabs, Hadamard test (complex amplitude), post-selected LCU step (with or without mid-circuit measurement); angle-independent compilation (`compile_fixed`) |
| `fejer.py` | Reconstruction of zeros from P(θ) alone: symmetric cosine fit, root finding, noise-floor handling, bootstrap, two-point sign test, non-Pauli diagnostic |
| `analysis.py` | Two-dimensional likelihood localisation of a zero from post-selected (LCU) data, with parametric bootstrap |
| `routing.py` | Routing on the IQM 54-qubit square-lattice model of `iqm-client` (`IQMFakeAphrodite`); checks that rings embed without SWAP gates |
| `simulate.py`, `emulate.py` | Qiskit Aer emulation with and without noise; emulator benchmark of every planned circuit family after routing |
| `budget.py`, `request.py` | Shots per planned point from its exact value; itemised QPU-time request |
| `export.py` | OpenQASM 3 for Euro-Q-Exa, native gate lists (RZ, R, RXX) for PIAST-Q, an IBM Runtime submission helper |
| `verify.py` | Verification suite V1–V8 (below) |
| `sweeps.py`, `plots.py`, `propfigs.py` | Parameter sweeps, plots, and the figures of the proposal and the feasibility proof |

## Installation

Python 3.12. Versions are pinned to the configuration that produced every number in the proposal.

```bash
python3 -m venv ~/venvs/isingzeros && source ~/venvs/isingzeros/bin/activate
pip install -r requirements.txt     # Qiskit 2.1.2, Qiskit Aer 0.17.2, iqm-client 36.0.0, NumPy 2.5.3, SciPy 1.18.1
python -m pytest -q tests           # 12 tests, about 1 minute
```

Or with conda: `conda env create -f environment.yml`.

The scripts in `slurm/` take the environment from `VENV`, for example `VENV=~/venvs/isingzeros sbatch slurm/verify.sbatch`. Run all commands from the repository root.

## Reproducing the numbers

Four commands with fixed random seeds reproduce every number in the proposal and the feasibility proof:

```bash
python -m isingzeros.routing --out results/routing.json   # routing on the 54-qubit model (~30 s)
sbatch slurm/emulate.sbatch                               # emulator benchmark (48 min on one 40-core node, 85 GB peak)
sbatch slurm/verify.sbatch                                # tests, V1-V8, routing, budget, report (about 1 h on 32 cores)
python -m isingzeros.request                              # itemised QPU request -> results/request.json
```

Figures: `python -m isingzeros.propfigs fig1`, `fig2` and `feas`. The figure text uses Liberation Sans if `ISINGZEROS_FONTDIR` points to its `.ttf` files, and DejaVu Sans otherwise.

## Verification suite

Generic noise model (not a device calibration): depolarising error p2 after every CZ, p2/10 after every physical single-qubit gate, virtual RZ without error, and 1 % symmetric readout error. Target: the zero of the 6 × 3 strip at (a0, θ0) = (+0.0213, 0.629), with φ = 0.6. Results are in `results/verify/`, summarised in `results/verify/REPORT.md`.

| Stage | What it checks | Result |
|:--|:--|:--|
| tests | Brute force = transfer matrix = statevector circuit = sampling; LCU for both signs; compilation independent of the angle; symmetries under Pauli noise | 12/12 pass |
| V1 | Exactness of the reconstruction without noise, 13 sizes up to 12 × 3 and two-leg ladders | Error ≤ 1.1 × 10⁻⁹ up to 8 × 4; 1.3 × 10⁻⁵ at 12 × 3 |
| V2 | Shot noise only, 40 angles × 5 × 10⁴ shots | rms 0.0011 in \|a\|, 0.0007 in θ; 95 % bootstrap coverage 0.93 / 0.97 |
| V3 | Circuit-level noise (density matrix): Pauli invariance, fit designs, noise floor, zero-noise extrapolation | Symmetries hold to 2 × 10⁻¹⁴; \|a0\| = 0.0219 / 0.0225 / 0.0242 / 0.0273 at p2 = 0.1 / 0.2 / 0.5 / 1 %, and 0.0213 / 0.0213 / 0.0210 / 0.0200 after extrapolation |
| V4 | Two-point sign test | 1.6 × 10⁴ shots per point give the correct sign with error probability 10⁻³ at p2 = 0.5 % |
| V5 | Measurability window: error against distance from the axis | rms ≈ 10⁻³ for \|a0\| ≤ 0.08 (6 × 3) |
| V6 | Reconstruction against direct LCU post-selection on the same zero | Median error 0.0013 / 0.0020 / 0.0040 / 0.0086 (2 × 10⁶ shots) against 0.0025 / 0.0067 / 0.018 / 0.036 (5 × 10⁶ shots) |
| V7 | Is mid-circuit reset needed under post-selection? | No: results with and without reset agree to 1.3 × 10⁻¹⁴ |
| V8 | Coverage of the LCU bootstrap intervals | Nominal without noise; under-covers with noise, so LCU positions need a systematic error from noise amplification |
| routing | Rings, slabs and LCU circuits on the 54-qubit model | Rings of 6–24 qubits need no SWAP gates; LCU 6 × 3: 123 CZ with mid-circuit measurement, 146 without |
| budget | Shots per planned point from exact values | 1.18 × 10⁷ shots for the base plan |

**Emulator benchmark** (`results/emulator.json`, `figs/feas_fig1.*`). One representative circuit of each of 15 planned families is routed and emulated:
- Without noise, every circuit reproduces the exact transfer-matrix value to 1.9 × 10⁻¹⁴.
- With gate noise, the surviving signal fraction F = (P_noisy − 2⁻ⁿ)/(P_ideal − 2⁻ⁿ) stays within about 10 % of the estimate (1 − p2)^CZ used by the budget.
- Time per noisy trajectory: under 1 ms up to 12 qubits; 0.7–1.2 s at 24 qubits; 4–5 s at 26–27 qubits on one 40-core node.

## Files in `results/` and `figs/`

- `results/verify/` — V1–V8 outputs and `REPORT.md`
- `results/routing.json` — gate counts and depths before and after routing
- `results/emulator.json` — emulator benchmark
- `results/budget.csv`, `results/request.json` — shot budget and QPU request
- `results/noisy_L8M4.json` — example noisy scan of the 8 × 4 ring
- `figs/prop_fig1.*`, `figs/prop_fig2.*`, `figs/feas_fig1.*` — figures with their source data (`*_data.json`)
- `figs/fr_window.png`, `figs/noisy_L8M4.png` — measurability window (V5), example noisy scan

## Hardware export

```bash
python -m isingzeros.export qasm  --L 8 --M 4 --theta 0.3:1.5:25 --phi 0.6 --outdir jobs/eqexa   # OpenQASM 3 for Euro-Q-Exa
python -m isingzeros.export piast --L 8 --M 3 --theta 0.3:1.5:13 --phi 0.6 --outdir jobs/piast   # native gate list for PIAST-Q
```

Circuits are compiled once from a template with symbolic angles, so every angle of a scan uses the same gate sequence. No credentials are stored in the package.

## Known limitations

- The dense transfer matrix reaches L·W ≤ 24 qubits. Larger rings need free fermions, and thicker slabs need tensor networks. The 32-qubit slab of the plan is extrapolated.
- Routing uses the `IQMFakeAphrodite` model shipped with `iqm-client`, not the real Euro-Q-Exa coupling map or calibration. The circuits will be re-compiled on the device.
- The generic noise model contains only Pauli noise, under which P(θ) keeps its symmetries exactly. On hardware, the asymmetric part of the fit measures non-Pauli errors, which no emulation can predict.
- A coherent error in the RZZ angle shifts all zeros in θ by about the same amount; it needs angle calibration.
- The success rate of post-selection falls exponentially with the distance from the axis. Points where it is too small are marked as unresolvable in the budget.

## Licence

MIT License, see `LICENSE`. Dependencies: Qiskit, Qiskit Aer and iqm-client (Apache 2.0); NumPy and SciPy (BSD 3-Clause).

## Author

Yue Fang, School of Electrical Engineering, Aalto University, Finland. Questions and bug reports: GitHub issues.
