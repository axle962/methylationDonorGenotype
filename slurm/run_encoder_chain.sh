#!/bin/bash
# Phase A ladder for another frozen encoder (a model bundle in the Hugging Face cache, e.g. your Perceiver),
# under its own output root so it can run next to the CpGPT results and be compared with them.
#   slurm/run_encoder_chain.sh <bundle> [<cpgpt-src>]      e.g. perceiver_large /scratch/am16489/CpGPT_AMG_Lab_perceiver_f349acb
# Root: outputs/<bundle>/ with encodings, predictor{,_seed1,_seed2}, experiments{,_seed1,_seed2}, decision.json.
# The beta-space control does not depend on the encoder (same sites, blocks and donor split), so
# <root>/predictor*_beta are links to the Phase A ones (outputs/predictor*_beta); run Phase A first.
# Not run here (10_decide reports them NOT EVALUATED): the unmasked leakage guard and the 5/20 Mb ablation.
#   01 (done) -> 02 GPU (--bundle) -> 03 seeds 0,1,2 (CPU) -> per seed 04 -> {06, 07} -> 08 -> 10
# Env: DRY_RUN=1 prints the commands; GPU_TIME_02 sets --time of the encoding job (default 06:00:00);
#      ENC02_CPU=1 encodes on a 24-CPU n2c48m24 node instead (CPU_TIME_02, default 12:00:00).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

bundle="${1:?usage: slurm/run_encoder_chain.sh <bundle> [<cpgpt-src>]}"
src="${2:-}"
root="outputs/$bundle"

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
pred() { [[ "$1" == 0 ]] && echo "$root/predictor" || echo "$root/predictor_seed$1"; }
expd() { [[ "$1" == 0 ]] && echo "$root/experiments" || echo "$root/experiments_seed$1"; }

for s in 0 1 2; do  # the encoder-independent beta-space control from Phase A
  phase_a=$([[ "$s" == 0 ]] && echo outputs/predictor_beta || echo "outputs/predictor_seed${s}_beta")
  [[ -d "$phase_a" ]] || { echo "error: $phase_a not found; run Phase A (03 --beta-space) first" >&2; exit 1; }
done
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  mkdir -p "$root"
  for s in 0 1 2; do
    link="$(pred "$s")_beta"
    [[ -e "$link" ]] || ln -s "$(realpath "$([[ "$s" == 0 ]] && echo outputs/predictor_beta || echo "outputs/predictor_seed${s}_beta")")" "$link"
  done
fi

enc_args=(--include-x --bundle "$bundle" --out "$root/encodings")
[[ -n "$src" ]] && enc_args+=(--cpgpt-src "$src")
if [[ "${ENC02_CPU:-0}" == 1 ]]; then  # encode on a whole CPU node (no checkpointing: generous --time)
  j02=$(submit "" slurm/02_encode.sbatch --partition=n2c48m24 --gres=none --mem=22G --cpus-per-task=24 \
    "--time=${CPU_TIME_02:-12:00:00}" "--job-name=jepa_02_encode_$bundle" -- "${enc_args[@]}" --allow-cpu)
else
  j02=$(submit "" slurm/02_encode.sbatch "--time=${GPU_TIME_02:-06:00:00}" "--job-name=jepa_02_encode_$bundle" -- "${enc_args[@]}")
fi
say "02_encode $bundle" "$j02"

all=()
declare -A j08
for s in 0 1 2; do
  j03=$(submit "$j02" slurm/03_train_cpu.sbatch -- --enc "$root/encodings" --out "$(pred "$s")" --seed "$s" --split-seed 0)
  j04=$(submit "$j03" slurm/04_evaluate.sbatch -- --encodings "$root/encodings" --pred "$(pred "$s")")
  j06=$(submit "$j04" slurm/06_covariates.sbatch -- --encodings "$root/encodings" --pred "$(pred "$s")" --out "$(expd "$s")/covariates")
  j07=$(submit "$j04" slurm/07_mqtl_deep.sbatch -- --encodings "$root/encodings" --pred "$(pred "$s")" --out "$(expd "$s")/mqtl_deep")
  dep08="$j06:$j07"; [[ "$s" != 0 ]] && dep08="$dep08:${j08[0]}"
  j08[$s]=$(submit "$dep08" slurm/08_structure.sbatch -- --encodings "$root/encodings" --pred "$(pred "$s")" --out "$(expd "$s")/structure" --train-seed "$s")
  say "seed $s" "03 $j03, 04 $j04, 06 $j06, 07 $j07, 08 ${j08[$s]}"
  all+=("$j06" "$j07" "${j08[$s]}")
done
j10=$(submit "$(IFS=:; echo "${all[*]}")" slurm/10_decide.sbatch -- --outputs "$root" --seeds 0,1,2)
say 10_decide "$j10"
