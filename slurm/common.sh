# Sourced at the top of every slurm/*.sbatch ON THE COMPUTE NODE.
# No module loads: as in predplan, everything runs inside the Singularity image with the overlay
# mounted read-only. The environment is the repo's uv venv (.venv, README: `uv sync --all-groups`),
# whose interpreter lives in the overlay, so python only works inside the container.
set -euo pipefail

: "${SLURM_SUBMIT_DIR:?not inside a Slurm job}"
REPO="$SLURM_SUBMIT_DIR"   # submit.sh always submits from the repo root
cd "$REPO"
set -a; source slurm/local.env; set +a
mkdir -p slurm/logs

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1   # the overlay's /root/.local has a broken pip; keep it off sys.path
export HF_HOME="$SCRATCH_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1     # model + dependencies are already in the cache
export TORCH_HOME="$SCRATCH_ROOT/cache/torch"
export TMPDIR="$SCRATCH_ROOT/tmp"
export AWS_EC2_METADATA_DISABLED=true AWS_MAX_ATTEMPTS=1   # CpGPTInferencer probes S3 on init; fail fast
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$TMPDIR"

# run_stage [--nv] <script.py> [args...]
#   Runs the script with .venv/bin/python inside the container and prints the sanity verdict the
#   script wrote to $SANITY_FILE as the LAST line of the log: "SANITY_OK" or "SANITY_FAIL: <reason>".
#   Exits non-zero on failure so --dependency=afterok chains stop.
run_stage() {
  local flags=(--fakeroot --writable-tmpfs)
  if [[ "${1:-}" == --nv ]]; then flags+=(--nv); shift; fi
  export SANITY_FILE="$REPO/slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.sanity"
  rm -f "$SANITY_FILE"
  echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), $(date -Is)"
  local rc=0
  singularity exec "${flags[@]}" --overlay "${EXT3_PATH}:ro" "$SIF_PATH" \
    /bin/bash -c 'cd "$1" && shift && exec .venv/bin/python "$@"' bash "$REPO" "$@" || rc=$?
  local verdict="SANITY_FAIL: script exited with code $rc without writing a sanity verdict"
  [[ -s "$SANITY_FILE" ]] && verdict="$(tail -n 1 "$SANITY_FILE")"
  if [[ $rc -ne 0 && "$verdict" == SANITY_OK ]]; then
    verdict="SANITY_FAIL: script exited with code $rc after its sanity checks"
  fi
  echo "$verdict"
  [[ "$verdict" == SANITY_OK ]]
}
