#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=n2c48m24
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=22G
#SBATCH --time=02:00:00
#SBATCH --job-name=jepa_06_covariates
#SBATCH --output=slurm/logs/%x_%j.out
# account comes from SBATCH_ACCOUNT at submit time (see submit.sh), never hardcoded
# Submit with: slurm/submit.sh slurm/06_covariates.sbatch [script args ...]
# Phase A covariate regression of the residual -> outputs/experiments/covariates/.

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
run_stage scripts/06_covariates.py "$@"
