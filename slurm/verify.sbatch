#!/bin/bash
#SBATCH --job-name=iz_verify
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=logs/verify_%j.out
##SBATCH --account=project_XXXXXXX      # LUMI / CSC: set your project
##SBATCH --partition=small               # LUMI: small ; Aalto Triton: batch-csl (or omit)
# Verification run V1-V8 (see README.md).  Usage:
#   sbatch slurm/verify.sbatch                 # everything (V1-V8, routing, budget, report)
#   sbatch slurm/verify.sbatch v3 v6           # only the listed stages
#   bash   slurm/verify.sbatch quick           # smoke test of every stage (~10-20 min on 4 cores)
module purge 2>/dev/null
# --- environment: either conda (environment.yml) or venv (requirements.txt) ---
[ -n "$VENV" ] && source "$VENV/bin/activate"     # e.g. VENV=$HOME/venvs/isingzeros sbatch slurm/verify.sbatch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONWARNINGS=ignore
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PYTHONPATH=$PWD:$PYTHONPATH
NP=${SLURM_CPUS_PER_TASK:-$(nproc)}
set -o pipefail
mkdir -p logs results/verify

STAGES="$@"
QUICK=""
if [ "$1" == "quick" ]; then QUICK="--quick"; STAGES="tests v1 v2 v7 v5 v3 v4 v6 v8 report"; OUT=results/verify_smoke; else OUT=results/verify; fi
[ -z "$STAGES" ] && STAGES="tests v1 v2 v7 v5 v3 v4 v6 v8 routing budget report"
echo "stages: $STAGES   nproc: $NP   outdir: $OUT   $(date)"
python -c "import numpy, scipy, qiskit, qiskit_aer; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'qiskit', qiskit.__version__, 'aer', qiskit_aer.__version__)" | tee logs/verify_versions.log

for s in $STAGES; do
  t0=$(date +%s)
  case $s in
    tests)   python -m pytest -q tests 2>&1 | tee logs/verify_tests.log ;;
    v[1-8])  python -m isingzeros.verify $s $QUICK --nproc $NP --outdir $OUT 2>&1 | tee logs/verify_$s.log ;;
    routing) python -m isingzeros.routing --out results/routing.json 2>&1 | tee logs/verify_routing.log ;;
    noisyfig) python -m isingzeros.sweeps noisy --L 8 --M 4 --phi 0.6 --ntheta 41 --shots 20000 --p2 0.005 --out results/noisy_L8M4.json 2>&1 | tee logs/verify_noisyfig.log ;;
    budget)  python -m isingzeros.budget --out results/budget.csv --nproc $(( NP < 16 ? NP : 16 )) 2>&1 | tee logs/verify_budget.log ;;
    report)  python -m isingzeros.verify report --outdir $OUT 2>&1 | tee logs/verify_report.log ;;
    *) echo "unknown stage $s"; exit 2 ;;
  esac
  echo "### stage $s finished with status $? after $(( $(date +%s) - t0 )) s"
done
echo "ALL DONE $(date)"
