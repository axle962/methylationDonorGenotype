#!/bin/bash
# Block until the given Slurm jobs leave the queue, or until MAX_WAIT seconds pass (default 540, so one
# call fits a 10-minute tool timeout), then print each job's state. Exit 0 = all finished, 2 = some
# still queued or running (call again).
#   slurm/wait_jobs.sh 39533 39540 ...
MAX_WAIT="${MAX_WAIT:-540}"
[[ $# -gt 0 ]] || { echo "usage: $0 JOBID..." >&2; exit 1; }
ids=$(IFS=,; echo "$*")
end=$((SECONDS + MAX_WAIT))
while (( SECONDS < end )); do
  [[ -z "$(squeue -h -j "$ids" 2>/dev/null)" ]] && break
  sleep 20
done
sacct -j "$ids" -X -n -o JobID,JobName%28,State,ExitCode,Elapsed
[[ -z "$(squeue -h -j "$ids" 2>/dev/null)" ]] && exit 0 || exit 2
