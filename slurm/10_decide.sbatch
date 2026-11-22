#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=n2c48m24
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=jepa_10_decide
#SBATCH --output=slurm/logs/%x_%j.out
# account comes from SBATCH_ACCOUNT at submit time (see submit.sh), never hardcoded
# Submit with: slurm/submit.sh slurm/10_decide.sbatch [script args ...]
# Applies docs/PREREG.md to the Phase A outputs -> outputs/decision.json.

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
run_stage scripts/10_decide.py "$@"
