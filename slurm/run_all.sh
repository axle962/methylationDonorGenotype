#!/bin/bash
# Submit the four stages chained with --dependency=afterok and print the job IDs.
# A stage that ends with SANITY_FAIL exits non-zero, so the later stages are cancelled.
# DRY_RUN=1 slurm/run_all.sh   prints the sbatch commands without submitting.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

prev=""
for stage in 01_prepare 02_encode 03_train 04_evaluate; do
  opts=(--parsable --kill-on-invalid-dep=yes)
  [[ -n "$prev" ]] && opts+=("--dependency=afterok:$prev")
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "slurm/submit.sh ${opts[*]} slurm/$stage.sbatch"
    prev="<$stage job id>"
    continue
  fi
  prev=$(slurm/submit.sh "${opts[@]}" "slurm/$stage.sbatch")
  prev="${prev%%;*}"
  echo "$stage: job $prev"
done
