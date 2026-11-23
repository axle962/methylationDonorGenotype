#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=c12m85-a100-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=44G
#SBATCH --time=12:00:00
#SBATCH --job-name=jepa_09_train_lewm
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x_%A_%a.out
# account comes from SBATCH_ACCOUNT at submit time (see submit.sh), never hardcoded
# Submit with: slurm/submit.sh slurm/09_train_lewm.sbatch [script args ...]
# Phase B LeWM block encoder, one lambda per array task (0.01, 0.1, 1) -> outputs/lewm/lam<lambda>/.

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
run_stage --nv scripts/09_train_lewm.py --lams 0.01,0.1,1 "$@"
