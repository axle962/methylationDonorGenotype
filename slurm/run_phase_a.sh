#!/bin/bash
# Phase A: 01 -> 02 -> 03 -> 04 -> {05, 06, 07, 08}, chained with --dependency=afterok; prints the job IDs.
# 02 runs with --include-x so 07 has its chrX positive control (chrX blocks are eval-only in 03).
# The 5/20 Mb encodings for 05 are a separate job: slurm/submit.sh slurm/02b_encode_sizes.sbatch (after 01);
# 05 skips the sizes whose encodings are absent and says so.
# DRY_RUN=1 slurm/run_phase_a.sh   prints the sbatch commands without submitting.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

submit() {  # submit <dependency job id or ""> <sbatch file> [script args...]
  local dep="$1" file="$2"; shift 2
  local opts=(--parsable --kill-on-invalid-dep=yes)
  [[ -n "$dep" ]] && opts+=("--dependency=afterok:$dep")
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "slurm/submit.sh ${opts[*]} $file $*" >&2
    echo "<$(basename "$file" .sbatch)>"
    return
  fi
  local id
  id=$(slurm/submit.sh "${opts[@]}" "$file" "$@")
  echo "${id%%;*}"
}

j01=$(submit "" slurm/01_prepare.sbatch);              echo "01_prepare:   $j01"
j02=$(submit "$j01" slurm/02_encode.sbatch --include-x); echo "02_encode:    $j02"
j03=$(submit "$j02" slurm/03_train.sbatch);            echo "03_train:     $j03"
j04=$(submit "$j03" slurm/04_evaluate.sbatch);         echo "04_evaluate:  $j04"
for stage in 05_ablations 06_covariates 07_mqtl_deep 08_structure; do
  j=$(submit "$j04" "slurm/$stage.sbatch"); printf '%-14s%s\n' "$stage:" "$j"
done
