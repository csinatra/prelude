#!/bin/bash
# Where the AIDE agent is right now. Run on the box during a run; no arguments.
#
# Two processes match the agent command line, and which one answers your
# question differs:
#   agent — AIDE itself: drafting code, calling the model, or blocked waiting
#           on the step to finish. A stack here in the Anthropic client is a
#           wiring problem.
#   exec  — the subprocess AIDE forks to run the code it just wrote. A stack
#           here is the agent's own solution running, however slowly, which is
#           a result rather than a fault.
# Diagnosing a quiet run means reading both, so this dumps both and labels
# which is which. That distinction is the whole point: a long step and a wedged
# one are identical in a log (see docs/RUNBOOK.md).
set -uo pipefail

mapfile -t PIDS < <(pgrep -f 'bin/aide data_dir')
if [ ${#PIDS[@]} -eq 0 ]; then
  echo "no aide process — no run in progress" >&2
  exit 1
fi

for pid in "${PIDS[@]}"; do
  ppid=$(ps -o ppid= -p "$pid" | tr -d ' ')
  # The exec subprocess is the one forked BY the other match, so a parent that
  # is itself in the list identifies it without depending on pid ordering.
  label="agent"
  for other in "${PIDS[@]}"; do
    [ "$ppid" = "$other" ] && label="exec "
  done
  printf -- '── %s  pid %s  %s elapsed  %s%% cpu\n' \
    "$label" "$pid" "$(ps -o etime= -p "$pid" | tr -d ' ')" \
    "$(ps -o pcpu= -p "$pid" | tr -d ' ')"
  sudo py-spy dump --pid "$pid" 2>&1 | sed -n '3,$p'
  echo
done

nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader \
  | sed 's/^/── gpu   /'
