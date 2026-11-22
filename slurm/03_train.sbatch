#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=g2-standard-12
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=44G
#SBATCH --time=03:00:00
#SBATCH --job-name=jepa_03_train
#SBATCH --output=slurm/logs/%x_%j.out
# account comes from SBATCH_ACCOUNT at submit time (see submit.sh), never hardcoded
# Submit with: slurm/submit.sh slurm/03_train.sbatch [script args ...]
# Latent block predictors (mlp, linear, block_mean) + residuals -> outputs/predictor/<kind>/.

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
run_stage --nv scripts/03_train_predictor.py "$@"
