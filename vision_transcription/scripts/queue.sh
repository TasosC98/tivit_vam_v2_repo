#!/usr/bin/env bash
# Train several models one after another, each followed by its evaluation, in the
# background (closing PuTTY is fine).
#
#   bash scripts/queue.sh <jobs file>
#   bash scripts/queue.sh scripts/jobs_review_B.txt
#
# Each line of the jobs file (blank lines and # comments are skipped) is one job:
#   <run name> <arch> <config> <evaluation name> <target> [config overrides...]
#   vam_strip_s2  strip  configs/vam_full.yaml  A1s2_vam_strip  full_f1  train.seed=1235
#
# For each job: train with scripts/run_experiment.sh (it resumes from last.pt if
# the run was interrupted), wait until the training ends, start the evaluation with
# scripts/evaluate_run.sh (it waits by itself while another evaluation uses the
# GPU), and go on with the next training at once. Start the queue again after a
# crash or a reboot: trainings whose train.log says "done." are not repeated, and
# evaluations whose results/<name>_test.txt is complete are not repeated either.
# A training that fails is reported in the log and the queue goes on.
#
# Optional: BASELINE_MIDI=<V2N's predicted_midi/pianovam_test> for every evaluation.
# Log: logs/queue_<jobs file name>.log
# Stop the queue: kill $(cat logs/queue_<jobs file name>.pid)
#   (the training that is running goes on; stop it too: kill $(cat runs/<run>/run.pid))
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

JOBS="${1:?usage: bash scripts/queue.sh <jobs file>}"
[ -f "$JOBS" ] || { echo "ERROR: no jobs file $JOBS"; exit 1; }
TAG="queue_$(basename "${JOBS%.*}")"
mkdir -p logs

if [ -z "${_QUEUE_CHILD:-}" ]; then
  if [ -f "logs/${TAG}.pid" ] && kill -0 "$(cat "logs/${TAG}.pid")" 2>/dev/null; then
    echo "ERROR: this queue is already running (pid $(cat "logs/${TAG}.pid")); log: logs/${TAG}.log"
    exit 1
  fi
  # check every line before starting, so a typo does not stop the queue days later
  while read -r name arch config ename target _; do
    case "$name" in ''|'#'*) continue;; esac
    [ -f "$config" ] || { echo "ERROR: job '$name': no config $config"; exit 1; }
    case "$arch" in strip|tiled) ;; *) echo "ERROR: job '$name': arch must be strip or tiled"; exit 1;; esac
    case "$target" in onset_f1|full_f1) ;; *) echo "ERROR: job '$name': target must be onset_f1 or full_f1"; exit 1;; esac
    [ -n "$ename" ] || { echo "ERROR: job '$name': no evaluation name"; exit 1; }
  done < "$JOBS"
  _QUEUE_CHILD=1 nohup bash "$SELF" "$@" >> "logs/${TAG}.log" 2>&1 &
  echo $! > "logs/${TAG}.pid"
  echo "queue started in the background (pid $!)"
  echo "  log:  tail -f logs/${TAG}.log"
  echo "  stop: kill \$(cat logs/${TAG}.pid)"
  exit 0
fi

say() { echo "=== $(date '+%Y-%m-%d %H:%M') $*"; }
say "queue ${JOBS} started"
# read the jobs on file descriptor 3, so that the commands below cannot eat lines
while read -r name arch config ename target rest <&3; do
  case "$name" in ''|'#'*) continue;; esac
  read -ra overrides <<< "${rest:-}"
  out="runs/${name}"

  if grep -q '^done\. best' "${out}/train.log" 2>/dev/null; then
    say "${name}: already trained"
  else
    if [ -f "${out}/run.pid" ] && kill -0 "$(cat "${out}/run.pid")" 2>/dev/null; then
      say "${name}: already running; waiting for it"
    else
      say "${name}: training (${arch}, ${config} ${rest:-})"
      if ! CONFIG="$config" bash scripts/run_experiment.sh "$name" "$arch" ${overrides[@]+"${overrides[@]}"}; then
        say "FAILED: ${name} did not start (see above); going on with the next job"
        continue
      fi
    fi
    while kill -0 "$(cat "${out}/run.pid")" 2>/dev/null; do sleep 60; done
    if ! grep -q '^done\. best' "${out}/train.log" 2>/dev/null; then
      say "FAILED: ${name} stopped before the end (see ${out}/train.log); going on with the next job"
      continue
    fi
    say "${name}: $(grep '^done\. best' "${out}/train.log" | tail -n 1)"
  fi

  if grep -q '=== mean over recordings' "results/${ename}_test.txt" 2>/dev/null; then
    say "${ename}: already evaluated"
  elif [ ! -f "${out}/best.pt" ]; then
    say "FAILED: no ${out}/best.pt, so no evaluation ${ename}"
  else
    say "${ename}: evaluation queued (log: logs/${ename}.log)"
    bash scripts/evaluate_run.sh "$ename" "$config" "${out}/best.pt" "$target"
  fi
done 3< "$JOBS"
say "queue ${JOBS}: all trainings done (the last evaluation may still be running: see logs/)"
