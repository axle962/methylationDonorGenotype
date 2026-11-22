#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=n2d32p
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=jepa_01_prepare
#SBATCH --output=slurm/logs/%x_%j.out
# account comes from SBATCH_ACCOUNT at submit time (see submit.sh), never hardcoded
# Submit with: slurm/submit.sh slurm/01_prepare.sbatch [script args ...]
# eGTEx betas -> arrow -> CpGPTDataSaver memmaps in data/processed/ (+ scripts/build_metadata.py).

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
run_stage scripts/01_prepare_egtex.py "$@"
