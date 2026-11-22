#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=n2c48m24
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --job-name=jepa_agent
#SBATCH --output=slurm/logs/%x_%j.out
# Headless Claude Code session that works through docs/agent/BRIEF.md, with a hard GPU-hour cap.
# CPU only: the session submits its own GPU jobs through slurm/submit.sh, whose guard refuses anything
# over GPU_BUDGET_HOURS; the watchdog below also cancels all jepa_* GPU jobs if real usage reaches it.
# Submit with: slurm/submit.sh slurm/agent_session.sbatch
# Transcript (stream-json): logs/agent/session_<jobid>.jsonl
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?not inside a Slurm job}"
cd "$SLURM_SUBMIT_DIR"
set -a; source slurm/local.env; set +a
source slurm/gpu_budget.sh
mkdir -p logs/agent slurm/logs
LOG="logs/agent/session_${SLURM_JOB_ID}.jsonl"
echo "agent session $SLURM_JOB_ID on $(hostname), $(date -Is); budget ${GPU_BUDGET_HOURS} GPU-h from ${GPU_BUDGET_START}"

watchdog() {
  while true; do
    read -r used committed < <(gpu_budget_usage)
    echo "$(date -Is) watchdog: GPU-h used ${used}, committed ${committed}" >> logs/agent/watchdog.log
    if awk -v u="$used" -v b="$GPU_BUDGET_HOURS" 'BEGIN { exit !(u >= b) }'; then
      echo "$(date -Is) BUDGET EXHAUSTED: ${used} >= ${GPU_BUDGET_HOURS}; cancelling jepa_* GPU jobs" >> logs/agent/watchdog.log
      touch logs/agent/BUDGET_EXHAUSTED
      squeue -u "$USER" -h -o "%i|%j|%b" | awk -F'|' '$2 ~ /^jepa_/ && $3 ~ /gpu/ { print $1 }' | xargs -r scancel
    fi
    sleep 300
  done
}
watchdog &
WD=$!
trap 'kill $WD 2>/dev/null || true' EXIT

PROMPT="You are the autonomous session described in docs/agent/BRIEF.md. The user, Aditya, authorised it on 2026-11-22: \
finish the tasks in that brief, submitting the CPU and GPU Slurm jobs they need through slurm/submit.sh, within the \
50 GPU-hour budget that slurm/gpu_budget.sh enforces. Read CLAUDE.md, docs/agent/BRIEF.md, docs/PREREG.md and \
docs/plan.md first, then work through the tasks in order, keeping docs/agent/REPORT.md current. Do not stop while your \
jobs are still running; stop only when the brief's tasks are done or one of its stopping conditions applies, and make \
the last section of the report say which."

rc=0
claude -p "$PROMPT" --permission-mode auto --output-format stream-json --verbose > "$LOG" 2> "logs/agent/session_${SLURM_JOB_ID}.err" || rc=$?
echo "claude exited with code $rc at $(date -Is)"
read -r used committed < <(gpu_budget_usage)
echo "final GPU-h used ${used}, committed ${committed} of ${GPU_BUDGET_HOURS}"
tail -c 2000 "$LOG" | tail -n 3 || true
exit $rc
