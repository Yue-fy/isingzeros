#!/bin/bash
# Larger chains (L up to 24, 2^24 complex vector = 256 MB) and ladders (2x12) with the transfer matrix only.
#SBATCH --job-name=isingzeros-bigL
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --output=logs/bigL_%j.out
export OMP_NUM_THREADS=1
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH
cd $SLURM_SUBMIT_DIR
NP=$SLURM_CPUS_PER_TASK
for cfg in "16 4 1" "18 4 1" "20 4 1" "12 4 2" "8 4 3"; do
  set -- $cfg
  python -m isingzeros.sweeps exact --L $1 --M $2 --W $3 --theta3 0.3 --ntheta 61 --nphi 61 --nproc $NP --out results/exact_L$1M$2W$3.json
done
