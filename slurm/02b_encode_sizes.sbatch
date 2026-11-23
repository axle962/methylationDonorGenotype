#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=c12m85-a100-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=44G
#SBATCH --time=06:00:00
#SBATCH --job-name=jepa_02b_encode_sizes
#SBATCH --array=0-1
#SBATCH --output=slurm/logs/%x_%A_%a.out
# account comes from SBATCH_ACCOUNT at submit time (see submit.sh), never hardcoded
# Submit with: slurm/submit.sh slurm/02b_encode_sizes.sbatch [script args ...]
# Frozen CpGPT block embeddings at 5 and 20 Mb (array task 0 -> 5 Mb, 1 -> 20 Mb) -> outputs/encodings_<N>mb/, for 05.

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
SIZES=(5 20)
MB=${SIZES[$SLURM_ARRAY_TASK_ID]}
run_stage --nv scripts/02_encode_blocks.py --window-mb "$MB" --out "outputs/encodings_${MB}mb" "$@"
