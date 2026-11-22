# Sourced at the top of every slurm/*.sbatch ON THE COMPUTE NODE.
# Ported from Torch/Singularity to NYU HPC Cloud Bursting (Nov 2026): no container; the interpreter is
# the CpGPT venv named by $VENV in slurm/local.env (default: the CpGPT_AMG_Lab checkout, whose cpgpt/
# is identical to origin/jepa_predictor's). Partitions are set per sbatch file.
set -euo pipefail

: "${SLURM_SUBMIT_DIR:?not inside a Slurm job}"
REPO="$SLURM_SUBMIT_DIR"   # submit.sh always submits from the repo root
cd "$REPO"
set -a; source slurm/local.env; set +a
mkdir -p slurm/logs

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/hf_cache}"
export HF_HUB_OFFLINE=1     # model + dependencies are already in the cache
export TORCH_HOME="$SCRATCH_ROOT/cache/torch"
export TMPDIR="$SCRATCH_ROOT/tmp/jepa"
export AWS_EC2_METADATA_DISABLED=true AWS_MAX_ATTEMPTS=1   # CpGPTInferencer probes S3 on init; fail fast
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$TMPDIR"

# run_stage [--nv] <script.py> [args...]
#   Runs the script with $VENV/bin/python and prints the sanity verdict the script wrote to
#   $SANITY_FILE as the LAST line of the log: "SANITY_OK" or "SANITY_FAIL: <reason>".
#   Exits non-zero on failure so --dependency=afterok chains stop. --nv is accepted and ignored
#   (it was the Singularity GPU flag); GPUs come from the partition's --gres.
run_stage() {
  if [[ "${1:-}" == --nv ]]; then shift; fi
  export SANITY_FILE="$REPO/slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.sanity"
  rm -f "$SANITY_FILE"
  echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), $(date -Is)"
  command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null || true
  local rc=0
  "$VENV/bin/python" "$@" || rc=$?
  local verdict="SANITY_FAIL: script exited with code $rc without writing a sanity verdict"
  [[ -s "$SANITY_FILE" ]] && verdict="$(tail -n 1 "$SANITY_FILE")"
  if [[ $rc -ne 0 && "$verdict" == SANITY_OK ]]; then
    verdict="SANITY_FAIL: script exited with code $rc after its sanity checks"
  fi
  echo "$verdict"
  [[ "$verdict" == SANITY_OK ]]
}
