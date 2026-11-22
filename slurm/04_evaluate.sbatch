#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=n2c48m24
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=22G
#SBATCH --time=02:00:00
#SBATCH --job-name=jepa_04_evaluate
#SBATCH --output=slurm/logs/%x_%j.out
# account comes from SBATCH_ACCOUNT at submit time (see submit.sh), never hardcoded
# Submit with: slurm/submit.sh slurm/04_evaluate.sbatch [script args ...]
# Donor matching + mQTL enrichment on test donors -> outputs/predictor/<kind>/evaluation.json.

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
run_stage scripts/04_evaluate.py "$@"
