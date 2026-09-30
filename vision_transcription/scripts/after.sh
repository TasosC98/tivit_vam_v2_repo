#!/usr/bin/env bash
# Run a command once a training run (or an evaluate_run.sh evaluation) has
# finished, e.g. so that two trainings do not share the GPU (each would run at
# about half speed). Waits in the background, so closing PuTTY is fine.
#
#   bash scripts/after.sh <run_or_evaluation_name> "<command>"
#   bash scripts/after.sh vam_full "CONFIG=configs/yt_full.yaml bash scripts/run_experiment.sh yt_full tiled"
#   bash scripts/after.sh E1_vam_full "DATA_CONFIG=... bash scripts/evaluate_run.sh E3a_zeroshot ..."
#
# Waits while runs/<name>/run.pid (a training) or logs/<name>.pid (an
# evaluation) is alive, then runs the command from the vision_transcription
# folder. Log: logs/after_<name>.log. Cancel the wait: kill $(cat logs/after_<name>.pid)
# Several commands can wait for the same run: the second one logs to
# logs/after_<name>_2.log (pid in logs/after_<name>_2.pid), the third to _3, ...
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

RUN="${1:?usage: bash scripts/after.sh <run_name> \"<command>\"}"
shift
CMD="$*"
[ -n "$CMD" ] || { echo "usage: bash scripts/after.sh <run_name> \"<command>\""; exit 1; }
PIDF="runs/${RUN}/run.pid"
[ -f "$PIDF" ] || PIDF="logs/${RUN}.pid"          # an evaluate_run.sh evaluation
mkdir -p logs

if [ -z "${_AFTER_CHILD:-}" ]; then
  [ -f "$PIDF" ] || { echo "ERROR: neither runs/${RUN}/run.pid nor logs/${RUN}.pid exists --"
                      echo "       is '${RUN}' the name of a training run or an evaluation?"; exit 1; }
  TAG="after_${RUN}"; k=1
  while [ -f "logs/${TAG}.pid" ] && kill -0 "$(cat "logs/${TAG}.pid")" 2>/dev/null; do
    k=$((k + 1)); TAG="after_${RUN}_${k}"        # that name belongs to a live waiter
  done
  _AFTER_CHILD=1 nohup bash "$SELF" "$RUN" "$CMD" > "logs/${TAG}.log" 2>&1 &
  echo $! > "logs/${TAG}.pid"
  echo "waiting for '${RUN}' to finish, then: ${CMD}"
  echo "  (background pid $!; log: logs/${TAG}.log; cancel: kill \$(cat logs/${TAG}.pid))"
  exit 0
fi

echo "=== $(date '+%Y-%m-%d %H:%M') waiting for '${RUN}' (pid $(cat "$PIDF"))"
while kill -0 "$(cat "$PIDF")" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%Y-%m-%d %H:%M') '${RUN}' finished; running: ${CMD}"
unset _AFTER_CHILD         # so an after.sh inside CMD queues its own waiter
bash -c "$CMD"
echo "=== $(date '+%Y-%m-%d %H:%M') done (exit $?)"
