#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=n2d32p
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=jepa_08_structure
#SBATCH --output=slurm/logs/%x_%j.out
# account comes from SBATCH_ACCOUNT at submit time (see submit.sh), never hardcoded
# Submit with: slurm/submit.sh slurm/08_structure.sbatch [script args ...]
# Phase A 3D-structure analysis (needs pyBigWig if the track is absent, and hic-straw) -> outputs/experiments/structure/.

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
run_stage scripts/08_structure.py "$@"
