#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=jepa_05_ablations
#SBATCH --output=slurm/logs/%x_%j.out
# account comes from SBATCH_ACCOUNT at submit time (see submit.sh), never hardcoded
# Submit with: slurm/submit.sh slurm/05_ablations.sbatch [script args ...]
# Phase A ablations (K sources, block size, tissue pairs, donor average) -> outputs/experiments/ablations/.

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
run_stage scripts/05_ablations.py "$@"
