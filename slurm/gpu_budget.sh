#!/bin/bash
# GPU-hour budget for this project's Slurm jobs (job names starting with jepa_), counted from
# GPU_BUDGET_START. Sourced by submit.sh and by the agent-session watchdog; also runnable directly:
#   slurm/gpu_budget.sh            prints used / committed / budget
# Settings (slurm/local.env): GPU_BUDGET_HOURS (default 50), GPU_BUDGET_START (YYYY-MM-DDTHH:MM:SS).
#   used      = GPU-hours already consumed (elapsed x GPUs), all states
#   committed = used + the not-yet-run remainder of each pending or running job's time limit
# A new GPU job is allowed only if committed + its own time limit x GPUs <= the budget.

: "${GPU_BUDGET_HOURS:=50}"
: "${GPU_BUDGET_START:=2026-11-22T00:00:00}"

# Time limit string (D-HH:MM:SS, HH:MM:SS, MM:SS, MM) -> hours
slurm_time_hours() {
  awk -v t="$1" 'BEGIN {
    d = 0; if (index(t, "-")) { split(t, a, "-"); d = a[1]; t = a[2] }
    n = split(t, p, ":"); h = 0; m = 0; s = 0
    if (n == 3) { h = p[1]; m = p[2]; s = p[3] } else if (n == 2) { m = p[1]; s = p[2] } else { m = p[1] }
    printf "%.4f\n", d * 24 + h + m / 60 + s / 3600 }'
}

# Prints "used committed" in GPU-hours for jepa_* jobs since GPU_BUDGET_START.
gpu_budget_usage() {
  sacct -u "$USER" -S "$GPU_BUDGET_START" -X -n -P -o JobName,State,ElapsedRaw,TimelimitRaw,AllocTRES,ReqTRES 2>/dev/null |
    awk -F'|' '
      $1 !~ /^jepa_/ { next }
      {
        tres = ($5 != "") ? $5 : $6; g = 0
        if (match(tres, /gres\/gpu=[0-9]+/)) g = substr(tres, RSTART + 9, RLENGTH - 9) + 0
        if (g == 0) next
        el = $3 / 3600; lim = ($4 ~ /^[0-9]+$/) ? $4 / 60 : el
        used += el * g
        if ($2 ~ /^(PENDING|RUNNING|REQUEUED|SUSPENDED|CONFIGURING)/) { rem = lim - el; if (rem > 0) com += rem * g }
      }
      END { printf "%.2f %.2f\n", used, used + com }'
}

# gpu_budget_check <sbatch args...>: exit 0 if the job requests no GPU or fits the budget.
gpu_budget_check() {
  local file="" a gres="" time="" args=("$@")
  for a in "${args[@]}"; do [[ "$a" == *.sbatch && -f "$a" ]] && { file="$a"; break; }; done
  for a in "${args[@]}"; do
    case "$a" in --gres=*) gres="${a#--gres=}" ;; --time=*|-t=*) time="${a#*=}" ;; esac
    [[ "$a" == "$file" ]] && break   # options after the script belong to the script
  done
  if [[ -n "$file" ]]; then
    [[ -z "$gres" ]] && gres=$(sed -n 's/^#SBATCH --gres=//p' "$file" | head -1)
    [[ -z "$time" ]] && time=$(sed -n 's/^#SBATCH --time=//p' "$file" | head -1)
  fi
  local ngpu
  ngpu=$(sed -n 's/.*gpu:\([a-z0-9]*:\)\{0,1\}\([0-9][0-9]*\).*/\2/p' <<<"$gres")
  [[ -z "$ngpu" || "$ngpu" == 0 ]] && return 0
  [[ -z "$time" ]] && { echo "gpu budget: refusing a GPU job with no --time" >&2; return 1; }
  local req used committed
  req=$(awk -v h="$(slurm_time_hours "$time")" -v g="$ngpu" 'BEGIN { printf "%.2f", h * g }')
  read -r used committed < <(gpu_budget_usage)
  if awk -v c="$committed" -v r="$req" -v b="$GPU_BUDGET_HOURS" 'BEGIN { exit !(c + r <= b) }'; then
    echo "gpu budget: ok, ${req} GPU-h requested; used ${used}, committed ${committed} of ${GPU_BUDGET_HOURS}" >&2
    return 0
  fi
  echo "gpu budget: REFUSED, ${req} GPU-h requested would exceed ${GPU_BUDGET_HOURS} (used ${used}, committed ${committed}). Lower --time or stop." >&2
  return 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && [[ -f slurm/local.env ]] && { set -a; source slurm/local.env; set +a; }
  read -r used committed < <(gpu_budget_usage)
  echo "jepa GPU-hours since ${GPU_BUDGET_START}: used ${used}, committed ${committed}, budget ${GPU_BUDGET_HOURS}"
fi
