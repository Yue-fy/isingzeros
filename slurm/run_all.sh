#!/bin/bash
# Full stage-1 workload on one CPU node.  Adjust --nproc to the node's cores.
# Usage: bash slurm/run_all.sh  (or submit slurm/stage1.sbatch)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:$PYTHONPATH
NP=${SLURM_CPUS_PER_TASK:-$(nproc)}
mkdir -p results figs logs

# 1. unit tests (cross-validation brute force / transfer matrix / circuits / shots)
python -m pytest -q tests | tee logs/tests.log

# 2. exact return-probability maps (transfer matrix, parallel) for the planned sizes
for cfg in "8 4 1" "10 4 1" "12 4 1" "14 4 1" "8 3 2" "10 3 2" "8 3 3"; do
  set -- $cfg
  python -m isingzeros.sweeps exact --L $1 --M $2 --W $3 --theta3 0.3 --ntheta 121 --nphi 121 --nproc $NP --out results/exact_L$1M$2W$3.json
done
python -m isingzeros.sweeps locus --ntheta 121 --nphi 121 --out results/locus.json

# 2b. aspect-ratio control (L ~ M) and finite-thickness comparison W = 1, 2, 3 at fixed L, M
for cfg in "4 4 1" "6 6 1" "8 8 1" "8 3 1" "8 3 2" "8 3 3"; do
  set -- $cfg
  python -m isingzeros.sweeps exact --L $1 --M $2 --W $3 --theta3 0.3 --ntheta 61 --nphi 61 --nproc $NP --out results/aspect_L$1M$2W$3.json
done

# 3. zeros in the complex-theta plane at three values of phi, chain and 2-leg ladder
for phi in 0.4 0.6 0.785398; do
  python -m isingzeros.sweeps zeros --L 8  --M 4 --phi $phi --nre 161 --nim 101 --out results/zeros_L8M4_phi$phi.json
  python -m isingzeros.sweeps zeros --L 12 --M 4 --phi $phi --nre 161 --nim 101 --out results/zeros_L12M4_phi$phi.json
  python -m isingzeros.sweeps zeros --L 6  --M 3 --W 2 --theta3 0.3 --phi $phi --nre 121 --nim 81 --out results/zeros_L6M3W2_phi$phi.json
done

# 4. noisy Aer simulations (two noise levels: superconducting-like and ion-like)
python -m isingzeros.sweeps noisy --L 8  --M 4 --phi 0.6 --ntheta 41 --shots 20000 --p2 0.005 --out results/noisy_L8M4_p2_5e-3.json
python -m isingzeros.sweeps noisy --L 8  --M 4 --phi 0.6 --ntheta 41 --shots 20000 --p2 0.002 --out results/noisy_L8M4_p2_2e-3.json
python -m isingzeros.sweeps noisy --L 12 --M 4 --phi 0.6 --ntheta 41 --shots 50000 --p2 0.005 --out results/noisy_L12M4_p2_5e-3.json
python -m isingzeros.sweeps noisy --L 6  --M 3 --W 2 --theta3 0.3 --phi 0.6 --ntheta 41 --shots 20000 --p2 0.005 --out results/noisy_L6M3W2_p2_5e-3.json

# 5. off-axis couplings via post-selected LCU
python -m isingzeros.sweeps lcu --L 6 --M 3 --phi 0.6 --theta 0.8 --a_zz="-0.3:0.3:13" --shots 200000 --p2 0     --out results/lcu_L6M3_ideal.json
python -m isingzeros.sweeps lcu --L 6 --M 3 --phi 0.6 --theta 0.8 --a_zz="-0.3:0.3:13" --shots 200000 --p2 0.002 --out results/lcu_L6M3_p2_2e-3.json
python -m isingzeros.sweeps lcu --L 4 --M 2 --phi 0.6 --theta 0.8 --a_zz="-0.2:0.2:5"  --shots 200000 --p2 0.005 --out results/lcu_L4M2_p2_5e-3.json

# 5b. routing on the IQM 54-qubit lattice target (needs iqm-client[qiskit]); must precede the budget,
#     which reads the routed CZ counts from results/routing.json
python -m isingzeros.routing --out results/routing.json

# 5c. 2D zero localisation (WP3 statistics): hardware-model demo (fast) and circuit-level-noise demo (density matrix, ~1 min)
python -m isingzeros.analysis demo --L 6 --M 3 --phi 0.6 --shots 200000 --halfwidth 0.03 --nboot 200 --out results/zero_localisation_demo_hw0.03.json
python -m isingzeros.analysis demo --L 6 --M 3 --phi 0.6 --shots 200000 --halfwidth 0.03 --nboot 100 --dm_p2 0.002 --out results/zero_localisation_noisy_L6M3_p2e-3.json

# 6. resource budget from exact probabilities at the planned points + routed CZ (budget.csv) + gate table (budget.json)
python -m isingzeros.budget --out results/budget.csv
python -m isingzeros.sweeps budget --out results/budget.json

# 7. figures
python -m isingzeros.plots exact results/exact_L8M4W1.json results/locus.json figs/exact_L8M4.png
python -m isingzeros.plots exact results/exact_L12M4W1.json results/locus.json figs/exact_L12M4.png
for f in results/zeros_*.json; do python -m isingzeros.plots zeros $f figs/$(basename ${f%.json}).png; done
for f in results/noisy_*.json; do python -m isingzeros.plots noisy $f figs/$(basename ${f%.json}).png; done
for f in results/lcu_*.json;   do python -m isingzeros.plots lcu   $f figs/$(basename ${f%.json}).png; done
python -m isingzeros.plots localisation results/zero_localisation_demo_hw0.03.json figs/zero2d_L6M3.png
python -m isingzeros.plots localisation results/zero_localisation_noisy_L6M3_p2e-3.json figs/zero2d_L6M3_noisy.png
echo "ALL DONE"
