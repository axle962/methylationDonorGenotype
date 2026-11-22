#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=n2c48m24
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=jepa_smoke_stage
#SBATCH --output=slurm/logs/%x_%j.out
# Re-run ONE smoke stage on CPU against the existing outputs/smoke artefacts (after a fix), instead of
# the whole smoke chain. Needs a prior slurm/smoke.sbatch run up to the stage before it.
# Submit with: slurm/submit.sh slurm/smoke_stage.sbatch scripts/09_train_lewm.py --smoke --encodings outputs/smoke/encodings --out-root outputs/smoke/lewm --lam 0.1

source "$SLURM_SUBMIT_DIR/slurm/common.sh"
run_stage "$@"
