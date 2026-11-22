#!/bin/bash
# Submit a job with the account from slurm/local.env exported into it (predplan convention:
# the account comes from SBATCH_ACCOUNT at submit time, never hardcoded in the sbatch files).
# Usage: slurm/submit.sh [--sbatch-option=value ...] <script.sbatch> [script args ...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"  # jobs cd to $SLURM_SUBMIT_DIR, i.e. the repo root

ENV_FILE="slurm/local.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "error: $ENV_FILE not found" >&2
  exit 1
fi
set -a; source "$ENV_FILE"; set +a

mkdir -p slurm/logs  # Slurm opens --output=slurm/logs/... before the job script runs
sbatch "$@"
