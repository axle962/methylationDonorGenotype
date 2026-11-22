#!/bin/bash
# Phase B: the LeWM lambda sweep (slurm/09_train_lewm.sbatch, array over 0.01, 0.1, 1) after stage 02.
# Usage: slurm/run_phase_b.sh [<02_encode job id>]
#   with a job id, 09 waits for that job (afterok); without one, outputs/encodings must already exist.
# Each array task also runs 03-07 on its own encodings. DRY_RUN=1 prints the sbatch command only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

opts=(--parsable --kill-on-invalid-dep=yes)
if [[ -n "${1:-}" ]]; then
  opts+=("--dependency=afterok:$1")
elif [[ ! -f outputs/encodings/Z.npy ]]; then
  echo "error: outputs/encodings/Z.npy not found; pass the 02_encode job id or run stage 02 first" >&2
  exit 1
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "slurm/submit.sh ${opts[*]} slurm/09_train_lewm.sbatch"
  exit 0
fi
j09=$(slurm/submit.sh "${opts[@]}" slurm/09_train_lewm.sbatch)
echo "09_train_lewm (array 0-2): ${j09%%;*}"
