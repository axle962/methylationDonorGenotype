#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=g2-standard-12
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=44G
#SBATCH --time=06:00:00
#SBATCH --job-name=jepa_02_encode
#SBATCH --output=slurm/logs/%x_%j.out
# account comes from SBATCH_ACCOUNT at submit time (see submit.sh), never hardcoded
# Submit with: slurm/submit.sh slurm/02_encode.sbatch [script args ...]
# Frozen CpGPT block embeddings -> outputs/encodings/.

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
run_stage --nv scripts/02_encode_blocks.py "$@"
