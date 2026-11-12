#!/usr/bin/env bash
# Train the KeyWise-hands runs one after another, each followed by its hand
# evaluation (scripts/evaluate_hands.sh), in the background (closing PuTTY is fine).
#
#   BASELINE_MIDI=/home/achatzigiannis/V2N/predicted_midi/pianovam_test \
#     bash scripts/queue_hands.sh scripts/jobs_hands.txt
#
# Each line of the jobs file (blank lines and # comments are skipped) is one job:
#   <run name> <config> <transcription evaluation name, or -> [config overrides...]
#   "-"    the transcription is frozen (its notes are E1 run 2's), so only the hands
#          are scored, with E1 run 2's thresholds (results/E1s2_vam_full_calibrate.txt)
#   a name the notes changed (joint fine-tuning): first calibrate + score the
#          transcription under that name (scripts/evaluate_run.sh, here in the queue),
#          then the hands with ITS thresholds and notes
#
# Start it again after a crash or a reboot: trainings whose train.log says "done."
# and hand evaluations whose results/<run>_hands_test.txt is complete are skipped.
# Log: logs/queue_<jobs file name>.log     Stop: kill $(cat logs/queue_<jobs file name>.pid)
#   (a training that is running goes on; stop it too: kill $(cat runs/<run>/run.pid))
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

JOBS="${1:?usage: bash scripts/queue_hands.sh <jobs file>}"
[ -f "$JOBS" ] || { echo "ERROR: no jobs file $JOBS"; exit 1; }
TAG="queue_$(basename "${JOBS%.*}")"
mkdir -p logs results

if [ -z "${_QUEUE_CHILD:-}" ]; then
  if [ -f "logs/${TAG}.pid" ] && kill -0 "$(cat "logs/${TAG}.pid")" 2>/dev/null; then
    echo "ERROR: this queue is already running (pid $(cat "logs/${TAG}.pid")); log: logs/${TAG}.log"
    exit 1
  fi
  # check every line before starting, so a typo does not stop the queue hours later
  while read -r name config ename _; do
    case "$name" in ''|'#'*) continue;; esac
    [ -f "$config" ] || { echo "ERROR: job '$name': no config $config"; exit 1; }
    [ -n "$ename" ] || { echo "ERROR: job '$name': third column must be an evaluation name or -"; exit 1; }
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
while read -r name config ename rest <&3; do
  case "$name" in ''|'#'*) continue;; esac
  read -ra overrides <<< "${rest:-}"
  out="runs/${name}"

  if grep -q '^done\. best' "${out}/train.log" 2>/dev/null; then
    say "${name}: already trained"
  else
    if [ -f "${out}/run.pid" ] && kill -0 "$(cat "${out}/run.pid")" 2>/dev/null; then
      say "${name}: already running; waiting for it"
    else
      say "${name}: training (${config} ${rest:-})"
      if ! CONFIG="$config" bash scripts/run_experiment.sh "$name" tiled ${overrides[@]+"${overrides[@]}"}; then
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

  if grep -q '=== given notes' "results/${name}_hands_test.txt" 2>/dev/null; then
    say "${name}: hands already evaluated"
    continue
  fi
  if [ ! -f "${out}/best.pt" ]; then
    say "FAILED: no ${out}/best.pt, so no evaluation"
    continue
  fi
  if [ "$ename" != "-" ]; then
    if ! grep -q '=== mean over recordings' "results/${ename}_test.txt" 2>/dev/null; then
      say "${ename}: transcription evaluation first (log: logs/${ename}.log)"
      # In the foreground (a separate bash process, so its GPU lock ends with it):
      # the hand evaluation below needs the thresholds it calibrates.
      if ! _EVAL_CHILD=1 bash scripts/evaluate_run.sh "$ename" "$config" "${out}/best.pt" full_f1 \
           > "logs/${ename}.log" 2>&1; then
        say "FAILED: ${ename} (see logs/${ename}.log); no hand evaluation for ${name}"
        continue
      fi
    fi
    say "${name}: hands evaluation queued, thresholds of ${ename} (log: logs/${name}_hands.log)"
    THRESHOLDS_FROM="results/${ename}_calibrate.txt" KEYWISE_MIDI="out/${ename}_test" \
      bash scripts/evaluate_hands.sh "$name" "$config" "${out}/best.pt"
  else
    say "${name}: hands evaluation queued (log: logs/${name}_hands.log)"
    bash scripts/evaluate_hands.sh "$name" "$config" "${out}/best.pt"
  fi
done 3< "$JOBS"
say "queue ${JOBS}: all trainings done (the last hand evaluation may still be running: see logs/)"
