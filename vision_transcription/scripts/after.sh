#!/usr/bin/env bash
# Run a command once a training run has finished, so that two trainings do not
# share the GPU (each would run at about half speed). Waits in the background,
# so closing PuTTY is fine.
#
#   bash scripts/after.sh <run_name> "<command>"
#   bash scripts/after.sh vam_full "CONFIG=configs/yt_full.yaml bash scripts/run_experiment.sh yt_full tiled"
#
# Waits while runs/<run_name>/run.pid is alive, then runs the command from the
# vision_transcription folder. Log: logs/after_<run_name>.log
# Cancel the wait: kill $(cat logs/after_<run_name>.pid)
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

RUN="${1:?usage: bash scripts/after.sh <run_name> \"<command>\"}"
shift
CMD="$*"
[ -n "$CMD" ] || { echo "usage: bash scripts/after.sh <run_name> \"<command>\""; exit 1; }
PIDF="runs/${RUN}/run.pid"
mkdir -p logs

if [ -z "${_AFTER_CHILD:-}" ]; then
  [ -f "$PIDF" ] || { echo "ERROR: no ${PIDF} -- is '${RUN}' the name of a run?"; exit 1; }
  _AFTER_CHILD=1 nohup bash "$SELF" "$RUN" "$CMD" > "logs/after_${RUN}.log" 2>&1 &
  echo $! > "logs/after_${RUN}.pid"
  echo "waiting for '${RUN}' to finish, then: ${CMD}"
  echo "  (background pid $!; log: logs/after_${RUN}.log; cancel: kill \$(cat logs/after_${RUN}.pid))"
  exit 0
fi

echo "=== $(date '+%Y-%m-%d %H:%M') waiting for '${RUN}' (pid $(cat "$PIDF"))"
while kill -0 "$(cat "$PIDF")" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%Y-%m-%d %H:%M') '${RUN}' finished; running: ${CMD}"
bash -c "$CMD"
echo "=== $(date '+%Y-%m-%d %H:%M') done (exit $?)"
