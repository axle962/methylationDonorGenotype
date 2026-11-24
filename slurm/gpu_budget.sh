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
  # sacct omits jobs still waiting on a dependency, so queued jepa_* jobs that squeue lists but
  # sacct does not are added to "committed" at their full time limit.
  {
    sacct -u "$USER" -S "$GPU_BUDGET_START" -X -n -P -o JobID,JobName,State,ElapsedRaw,TimelimitRaw,AllocTRES,ReqTRES 2>/dev/null |
      sed 's/^/A|/'
    squeue -r -u "$USER" -h -o "%i|%j|%l|%b" 2>/dev/null | while IFS='|' read -r id name lim gres; do
      echo "Q|${id}|${name}|$(slurm_time_hours "$lim")|${gres}"
    done
  } | awk -F'|' '
      $3 !~ /^jepa_/ { next }
      $1 == "A" && $2 ~ /\[/ { next }  # a pending array summary row; squeue -r lists its tasks
      $1 == "A" {
        seen[$2] = 1
        tres = ($7 != "") ? $7 : $8; g = 0
        if (match(tres, /gres\/gpu=[0-9]+/)) g = substr(tres, RSTART + 9, RLENGTH - 9) + 0
        if (g == 0) next
        el = $5 / 3600; lim = ($6 ~ /^[0-9]+$/) ? $6 / 60 : el
        used += el * g
        if ($4 ~ /^(PENDING|RUNNING|REQUEUED|SUSPENDED|CONFIGURING)/) { rem = lim - el; if (rem > 0) com += rem * g }
        next
      }
      $1 == "Q" { q[$2] = $0 }
      END {
        for (id in q) {
          if (id in seen) continue
          split(q[id], f, "|"); g = 0
          if (match(f[5], /gpu:([a-z0-9]+:)?[0-9]+/)) { s = substr(f[5], RSTART, RLENGTH); n = split(s, p, ":"); g = p[n] + 0 }
          com += f[4] * g
        }
        printf "%.2f %.2f\n", used, used + com
      }'
}

# gpu_budget_check <sbatch args...>: exit 0 if the job requests no GPU or fits the budget.
gpu_budget_check() {
  local file="" a gres="" time="" array="" args=("$@")
  for a in "${args[@]}"; do [[ "$a" == *.sbatch && -f "$a" ]] && { file="$a"; break; }; done
  for a in "${args[@]}"; do
    case "$a" in --gres=*) gres="${a#--gres=}" ;; --time=*|-t=*) time="${a#*=}" ;; --array=*) array="${a#--array=}" ;; esac
    [[ "$a" == "$file" ]] && break   # options after the script belong to the script
  done
  if [[ -n "$file" ]]; then
    [[ -z "$gres" ]] && gres=$(sed -n 's/^#SBATCH --gres=//p' "$file" | head -1)
    [[ -z "$time" ]] && time=$(sed -n 's/^#SBATCH --time=//p' "$file" | head -1)
    [[ -z "$array" ]] && array=$(sed -n 's/^#SBATCH --array=//p' "$file" | head -1)
  fi
  # array tasks: "0-2", "0,3,5-7" (a %throttle suffix is ignored); each task gets the full time limit
  local ntask=1
  [[ -n "$array" ]] && ntask=$(awk -v a="${array%%%*}" 'BEGIN { n = 0; k = split(a, r, ",")
      for (i = 1; i <= k; i++) { if (split(r[i], e, "-") == 2) n += e[2] - e[1] + 1; else n += 1 } print n }')
  local ngpu
  ngpu=$(sed -n 's/.*gpu:\([a-z0-9]*:\)\{0,1\}\([0-9][0-9]*\).*/\2/p' <<<"$gres")
  [[ -z "$ngpu" || "$ngpu" == 0 ]] && return 0
  [[ -z "$time" ]] && { echo "gpu budget: refusing a GPU job with no --time" >&2; return 1; }
  local req used committed
  req=$(awk -v h="$(slurm_time_hours "$time")" -v g="$ngpu" -v n="$ntask" 'BEGIN { printf "%.2f", h * g * n }')
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
