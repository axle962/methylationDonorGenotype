#!/bin/bash
# Phase A full chain (docs/agent/BRIEF.md task 3), --dependency=afterok throughout; prints the job IDs.
#   01 -> 02 masked (--include-x) -> 03 seeds 0,1,2 (CPU)          seed 0 -> outputs/predictor, s -> outputs/predictor_seed<s>
#                                 -> 03 --beta-space seed 0 -> seeds 1,2 (shared outputs/encodings_beta, built by seed 0)
#         per seed: 04 --beta-space -> 04 -> {06, 07} -> 08 (08 rewrites <pred>/metrics.json and splits.json, so it runs
#                   after 06/07 of its seed; seeds 1,2 wait for seed 0's 08 so the Hi-C cache is written once)
#   after 02 masked (so its run time can be measured and the others' --time tightened before they start):
#         02 unmasked (--probe-mask none) -> 03 -> 04 on outputs/{encodings,predictor}_unmasked
#         02 at 5 Mb and 20 Mb -> 05 (seed 0, after 04 seed 0)
#   10 after everything.
# Stage outputs: experiments for seed 0 in outputs/experiments/<name>, for seed s in outputs/experiments_seed<s>/<name>.
# Env: DRY_RUN=1 prints the commands without submitting; SKIP_01=1 starts from existing data/processed (DEP_01=<jobid> makes 02 wait for a stage 01 job submitted separately);
#      ENC02_CPU=1 runs the main 02 on a 24-CPU n2c48m24 node (CPU_TIME_02, default 3 days);
#      GPU_TIME_02, GPU_TIME_02U, GPU_TIME_02_5, GPU_TIME_02_20 set --time of the GPU jobs (default 06:00:00).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

submit() {  # submit <colon-separated dependency ids or ""> <sbatch file> [sbatch options before the file...] -- [script args...]
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
pred() { [[ "$1" == 0 ]] && echo outputs/predictor || echo "outputs/predictor_seed$1"; }
expd() { [[ "$1" == 0 ]] && echo outputs/experiments || echo "outputs/experiments_seed$1"; }

j01="${DEP_01:-}"  # with SKIP_01=1: an already-submitted stage 01 job that 02 waits for
if [[ "${SKIP_01:-0}" != 1 ]]; then j01=$(submit "" slurm/01_prepare.sbatch); say 01_prepare "$j01"; fi
if [[ "${ENC02_CPU:-0}" == 1 ]]; then  # main encoding on a whole CPU node (no checkpointing: generous --time)
  j02=$(submit "$j01" slurm/02_encode.sbatch --partition=n2c48m24 --gres=none --mem=22G --cpus-per-task=24 \
    "--time=${CPU_TIME_02:-3-00:00:00}" -- --include-x --allow-cpu --out outputs/encodings)
else
  j02=$(submit "$j01" slurm/02_encode.sbatch "--time=${GPU_TIME_02:-06:00:00}" -- --include-x --out outputs/encodings)
fi
say 02_encode "$j02"

# extra encodings: after 02 masked
j02u=$(submit "$j02" slurm/02_encode.sbatch "--time=${GPU_TIME_02U:-06:00:00}" --job-name=jepa_02_encode_unmasked -- \
  --probe-mask none --out outputs/encodings_unmasked)
say 02_encode_unmasked "$j02u"
j02_5=$(submit "$j02" slurm/02_encode.sbatch "--time=${GPU_TIME_02_5:-06:00:00}" --job-name=jepa_02_encode_5mb -- \
  --window-mb 5 --out outputs/encodings_5mb)
say 02_encode_5mb "$j02_5"
j02_20=$(submit "$j02" slurm/02_encode.sbatch "--time=${GPU_TIME_02_20:-06:00:00}" --job-name=jepa_02_encode_20mb -- \
  --window-mb 20 --out outputs/encodings_20mb)
say 02_encode_20mb "$j02_20"

all=()
declare -A j03 j03b j04 j08
for s in 0 1 2; do
  j03[$s]=$(submit "$j02" slurm/03_train_cpu.sbatch -- --enc outputs/encodings --out "$(pred $s)" --seed "$s" --split-seed 0)
  say "03_train seed$s" "${j03[$s]}"
done
j03b[0]=$(submit "$j02" slurm/03_train_cpu.sbatch --job-name=jepa_03_beta -- --beta-space --enc outputs/encodings --out "$(pred 0)" --seed 0 --split-seed 0)
say "03_beta seed0" "${j03b[0]}"
for s in 1 2; do
  j03b[$s]=$(submit "${j03b[0]}" slurm/03_train_cpu.sbatch --job-name=jepa_03_beta -- --beta-space --enc outputs/encodings --out "$(pred $s)" --seed "$s" --split-seed 0)
  say "03_beta seed$s" "${j03b[$s]}"
done
for s in 0 1 2; do
  jb=$(submit "${j03b[$s]}" slurm/04_evaluate.sbatch --job-name=jepa_04_beta -- --beta-space --encodings outputs/encodings --pred "$(pred $s)")
  say "04_beta seed$s" "$jb"
  j04[$s]=$(submit "${j03[$s]}:$jb" slurm/04_evaluate.sbatch -- --encodings outputs/encodings --pred "$(pred $s)")
  say "04_evaluate seed$s" "${j04[$s]}"
  j06=$(submit "${j04[$s]}" slurm/06_covariates.sbatch -- --encodings outputs/encodings --pred "$(pred $s)" --out "$(expd $s)/covariates")
  j07=$(submit "${j04[$s]}" slurm/07_mqtl_deep.sbatch -- --encodings outputs/encodings --pred "$(pred $s)" --out "$(expd $s)/mqtl_deep")
  say "06_covariates seed$s" "$j06"; say "07_mqtl_deep seed$s" "$j07"
  dep08="$j06:$j07"; [[ "$s" != 0 ]] && dep08="$dep08:${j08[0]}"
  j08[$s]=$(submit "$dep08" slurm/08_structure.sbatch -- --encodings outputs/encodings --pred "$(pred $s)" --out "$(expd $s)/structure" --train-seed "$s")
  say "08_structure seed$s" "${j08[$s]}"
  all+=("$j06" "$j07" "${j08[$s]}")
done

j03u=$(submit "$j02u" slurm/03_train_cpu.sbatch -- --enc outputs/encodings_unmasked --out outputs/predictor_unmasked --seed 0 --split-seed 0)
j04u=$(submit "$j03u" slurm/04_evaluate.sbatch -- --encodings outputs/encodings_unmasked --pred outputs/predictor_unmasked)
say 03_train_unmasked "$j03u"; say 04_evaluate_unmasked "$j04u"
j05=$(submit "${j04[0]}:$j02_5:$j02_20" slurm/05_ablations.sbatch -- --encodings outputs/encodings --pred outputs/predictor \
  --out outputs/experiments/ablations --size-encodings "outputs/encodings_{mb}mb")
say 05_ablations "$j05"
all+=("$j04u" "$j05")
j10=$(submit "$(IFS=:; echo "${all[*]}")" slurm/10_decide.sbatch -- --outputs outputs --seeds 0,1,2)
say 10_decide "$j10"
