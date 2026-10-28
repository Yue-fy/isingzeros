#!/bin/bash
#SBATCH --job-name=isingzeros
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=logs/stage1_%j.out
##SBATCH --account=project_XXXXXXX      # LUMI / CSC: set your project
##SBATCH --partition=small               # LUMI: small / standard ; Triton (Aalto): batch-csl
module purge
# --- option A: conda/mamba environment created from environment.yml ---
# module load  Anaconda3 ; source activate isingzeros
# --- option B: python venv ---
# python3 -m venv $HOME/venvs/isingzeros && source $HOME/venvs/isingzeros/bin/activate && pip install -r requirements.txt
export OMP_NUM_THREADS=1
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH
cd $SLURM_SUBMIT_DIR
bash slurm/run_all.sh
