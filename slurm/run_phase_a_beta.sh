#!/bin/bash
# Beta-space ladder (docs/PREREG.md amendment A1), all CPU, --dependency=afterok throughout; prints the job IDs.
# Needs the Phase A outputs of slurm/run_phase_a.sh: outputs/encodings_beta, outputs/predictor{,_seed1,_seed2}_beta
# (03/04 --beta-space) and outputs/encodings_unmasked (02 --probe-mask none).
#   unmasked: 03 --beta-space on outputs/encodings_unmasked -> 04 --beta-space   (leakage guard, seed 0)
#   per seed: {06, 07} on the beta-space residual -> 08 (seeds 1,2 wait for seed 0's 08: one Hi-C cache)
#   10 --space beta after everything -> outputs/decision_beta.json
# Stage outputs: outputs/experiments_beta/<name> for seed 0, outputs/experiments_seed<s>_beta/<name> otherwise.
# Env: DRY_RUN=1 prints the commands without submitting.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

submit() {  # submit <colon-separated dependency ids or ""> <sbatch file> [sbatch options...] -- [script args...]
  local dep="$1" file="$2"; shift 2
  local opts=(--parsable --kill-on-invalid-dep=yes)
  while [[ $# -gt 0 && "$1" != -- ]]; do opts+=("$1"); shift; done
  [[ "${1:-}" == -- ]] && shift
  [[ -n "$dep" ]] && opts+=("--dependency=afterok:$dep")
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "slurm/submit.sh ${opts[*]} $file $*" >&2
    echo "<$(basename "$file" .sbatch)_$RANDOM>"
    return
  fi
  local id
  id=$(slurm/submit.sh "${opts[@]}" "$file" "$@")
  echo "${id%%;*}"
}
say() { printf '%-22s%s\n' "$1:" "$2"; }
predb() { [[ "$1" == 0 ]] && echo outputs/predictor_beta || echo "outputs/predictor_seed$1_beta"; }
expb() { [[ "$1" == 0 ]] && echo outputs/experiments_beta || echo "outputs/experiments_seed$1_beta"; }

for need in outputs/encodings_beta/Z.npy outputs/encodings_unmasked/Z.npy; do
  [[ -f "$need" || "${DRY_RUN:-0}" == 1 ]] || { echo "missing $need: run slurm/run_phase_a.sh first" >&2; exit 1; }
done
for s in 0 1 2; do
  [[ -f "$(predb $s)/mlp_joint/R.npy" || "${DRY_RUN:-0}" == 1 ]] || { echo "missing $(predb $s)/mlp_joint/R.npy" >&2; exit 1; }
done

all=()
j03u=$(submit "" slurm/03_train_cpu.sbatch --job-name=jepa_03_beta_unmasked -- \
  --beta-space --enc outputs/encodings_unmasked --out outputs/predictor_unmasked --seed 0 --split-seed 0)
j04u=$(submit "$j03u" slurm/04_evaluate.sbatch --job-name=jepa_04_beta_unmasked -- \
  --beta-space --encodings outputs/encodings_unmasked --pred outputs/predictor_unmasked)
say 03_beta_unmasked "$j03u"; say 04_beta_unmasked "$j04u"
all+=("$j04u")

j08_0=""
for s in 0 1 2; do
  j06=$(submit "" slurm/06_covariates.sbatch --job-name=jepa_06_beta -- \
    --encodings outputs/encodings_beta --pred "$(predb $s)" --out "$(expb $s)/covariates")
  j07=$(submit "" slurm/07_mqtl_deep.sbatch --job-name=jepa_07_beta -- \
    --encodings outputs/encodings_beta --pred "$(predb $s)" --out "$(expb $s)/mqtl_deep")
  say "06_beta seed$s" "$j06"; say "07_beta seed$s" "$j07"
  dep08="$j06:$j07"; [[ "$s" != 0 ]] && dep08="$dep08:$j08_0"
  j08=$(submit "$dep08" slurm/08_structure.sbatch --job-name=jepa_08_beta -- \
    --encodings outputs/encodings_beta --pred "$(predb $s)" --out "$(expb $s)/structure" --train-seed "$s")
  say "08_beta seed$s" "$j08"
  [[ "$s" == 0 ]] && j08_0="$j08"
  all+=("$j06" "$j07" "$j08")
done

j10=$(submit "$(IFS=:; echo "${all[*]}")" slurm/10_decide.sbatch --job-name=jepa_10_decide_beta -- \
  --outputs outputs --seeds 0,1,2 --space beta)
say 10_decide_beta "$j10"
