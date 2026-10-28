#!/bin/bash
#SBATCH --job-name=iz_emulate
#SBATCH --time=08:00:00
#SBATCH --partition=batch-csl            # Aalto Triton; change for your cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=120G
#SBATCH --exclusive
#SBATCH --output=logs/emulate_%j.out
# Emulator benchmark of the planned Euro-Q-Exa circuits on one full CPU node (feasibility proof; see README.md).
#   sbatch slurm/emulate.sbatch
module purge 2>/dev/null
[ -n "$VENV" ] && source "$VENV/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONWARNINGS=ignore
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH=$PWD:$PYTHONPATH
PY=${PY:-python}
set -o pipefail
mkdir -p logs results
$PY -m isingzeros.emulate --out results/emulator.json "$@" 2>&1 | grep -v Warning | tee logs/emulate.log
