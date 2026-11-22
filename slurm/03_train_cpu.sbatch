#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=n2d32p
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=jepa_03_train_cpu
#SBATCH --output=slurm/logs/%x_%j.out
# account comes from SBATCH_ACCOUNT at submit time (see submit.sh), never hardcoded
# Submit with: slurm/submit.sh slurm/03_train_cpu.sbatch [script args ...]
# Stage 03 on CPU (the predictors are small: ~45k training pairs per epoch), so the GPU budget goes to stage 02.

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
run_stage scripts/03_train_predictor.py --allow-cpu "$@"
