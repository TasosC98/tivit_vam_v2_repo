#!/usr/bin/env bash
# Phase 3 of EXPERIMENT_PLAN.md for one trained model, in the background:
# calibrate the decode thresholds on VALID, then score TEST with exactly those
# thresholds (no copying numbers by hand).
#
#   bash scripts/evaluate_run.sh <name> <config> <checkpoint> <target>
#     <name>    label for the outputs, e.g. E1_vam_full
#     <target>  full_f1  (PianoVAM: key releases are real)
#               onset_f1 (PianoYT: offsets include the sustain pedal)
#
# Optional environment variables:
#   DATA_CONFIG=configs/yt_full.yaml   evaluate on the OTHER dataset (cross-dataset);
#                                      calibration then uses that dataset's valid split
#   THRESHOLDS="0.60 0.50"             skip calibration, use these (onset frame)
#   THRESHOLDS_FROM=results/E1_vam_full_calibrate.txt
#                                      skip calibration, reuse an earlier run's thresholds
#   BASELINE_MIDI=~/V2N/predicted_midi/pianovam_test   also compare with another
#                                      system's predictions (score_midi, paired test)
#   TIMING=1                           also tune onset timing on valid (sub-frame
#                                      onsets, a time shift, an extra shift for black keys)
#   TIMING_GRID=-0.05:0.20:0.01        time shifts to try (s); widen for cross-dataset
#
# Evaluations run one at a time: start several and they queue up by themselves.
# The model's outputs are kept in rolls/<name>/ (not downloaded with out/), so
# running the same <name> again (e.g. with TIMING=1) decodes them without the GPU.
#
# Outputs: logs/<name>.log (watch this), results/<name>_calibrate.txt,
#          results/<name>_test.txt + .csv, out/<name>_test/*.mid
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

if [ $# -ne 4 ]; then
  sed -n '2,28p' "$SELF"; exit 1
fi
NAME="$1"; CONFIG="$2"; CKPT="$3"; TARGET="$4"
[ -f "$CONFIG" ] || { echo "ERROR: no config $CONFIG"; exit 1; }
[ -f "$CKPT" ] || { echo "ERROR: no checkpoint $CKPT"; exit 1; }
case "$TARGET" in onset_f1|full_f1) ;; *) echo "ERROR: target must be onset_f1 or full_f1"; exit 1;; esac
if [ -n "${THRESHOLDS_FROM:-}" ] && [ ! -f "$THRESHOLDS_FROM" ]; then
  echo "ERROR: THRESHOLDS_FROM file not found: $THRESHOLDS_FROM"; exit 1
fi
mkdir -p logs results out

# The settings line calibrate prints: "decode.onset_threshold=X decode.frame_threshold=Y"
# (+ decode.onset_timing / time_shift_s / black_shift_s when calibrated with TIMING=1).
decode_line_in() {
  grep -E '^[[:space:]]*decode\.onset_threshold=' "$1" | tail -n 1 | sed 's/^[[:space:]]*//'
}

PIDFILE="logs/${NAME}.pid"
if [ -z "${_EVAL_CHILD:-}" ]; then
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "ERROR: evaluation '${NAME}' is already running (pid $(cat "$PIDFILE"))."
    echo "       Watch it: tail -f logs/${NAME}.log   (or pick another name)"
    exit 1
  fi
  _EVAL_CHILD=1 nohup bash "$SELF" "$@" > "logs/${NAME}.log" 2>&1 &
  echo $! > "$PIDFILE"
  echo "evaluation '${NAME}' started in the background (pid $!)."
  echo "  watch: tail -f logs/${NAME}.log"
  exit 0
fi

export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-2048}"
export PYTHONUNBUFFERED=1                  # logs show progress as it happens
DC=()
[ -n "${DATA_CONFIG:-}" ] && DC=(--data_config "$DATA_CONFIG")
echo "=== $(date '+%Y-%m-%d %H:%M') ${NAME}: ${CKPT}" \
     "${DATA_CONFIG:+on the dataset of $DATA_CONFIG}"

# One evaluation at a time: each needs up to ~11 GB of GPU memory, so two of
# them next to a training run do not fit in 24 GB. An evaluation started while
# another one runs waits here and starts by itself when the other one ends (the
# lock is released when that process exits, even if it fails).
if command -v flock >/dev/null 2>&1; then
  exec 9> logs/.evaluate.lock
  if ! flock -n 9; then
    echo "--- $(date '+%Y-%m-%d %H:%M') waiting: another evaluation is using the GPU"
    flock 9
    echo "--- $(date '+%Y-%m-%d %H:%M') the other evaluation finished; starting"
  fi
fi

if [ -n "${THRESHOLDS:-}" ]; then
  read -r OT FT <<< "$THRESHOLDS"
  DECODE="decode.onset_threshold=${OT} decode.frame_threshold=${FT}"
  echo "using the given thresholds: onset=${OT} frame=${FT}"
else
  if [ -n "${THRESHOLDS_FROM:-}" ]; then
    src="$THRESHOLDS_FROM"
    echo "--- reusing the thresholds calibrated in ${src}"
  else
    src="results/${NAME}_calibrate.txt"
    echo "--- calibrating on valid (target ${TARGET}) -> ${src}"
    TM=()
    [ -n "${TIMING:-}" ] && TM=(--timing)
    [ -n "${TIMING_GRID:-}" ] && TM+=("--shift_grid=${TIMING_GRID}")
    if ! python -m pianovam_vision.calibrate --config "$CONFIG" --checkpoint "$CKPT" \
         --split valid --target "$TARGET" ${DC[@]+"${DC[@]}"} ${TM[@]+"${TM[@]}"} \
         --rolls_dir "rolls/${NAME}/valid" > "$src" 2>&1; then
      echo "FAILED: calibration -- see ${src}"; exit 1
    fi
    grep -E "NOTE|Note F1|best timing" "$src"
  fi
  DECODE="$(decode_line_in "$src")"
  [ -n "$DECODE" ] || { echo "FAILED: no thresholds found in ${src}"; exit 1; }
  OT="${DECODE#decode.onset_threshold=}"; OT="${OT%% *}"
  FT="${DECODE#*decode.frame_threshold=}"; FT="${FT%% *}"
fi

echo "--- scoring test with onset=${OT} frame=${FT} -> results/${NAME}_test.txt"
[ "$DECODE" = "decode.onset_threshold=${OT} decode.frame_threshold=${FT}" ] \
  || echo "    decode settings: ${DECODE}"
# DECODE is a list of key=value overrides: split it into words on purpose.
# shellcheck disable=SC2086
if ! python -m pianovam_vision.evaluate --config "$CONFIG" --checkpoint "$CKPT" \
     --split test ${DC[@]+"${DC[@]}"} --csv "results/${NAME}_test.csv" \
     --save_midi "out/${NAME}_test" --rolls_dir "rolls/${NAME}/test" \
     $DECODE > "results/${NAME}_test.txt" 2>&1; then
  echo "FAILED: evaluation -- see results/${NAME}_test.txt"; exit 1
fi
sed -n '/=== mean over recordings/,$p' "results/${NAME}_test.txt"

if [ -n "${BASELINE_MIDI:-}" ]; then
  echo "--- comparing with ${BASELINE_MIDI} -> results/${NAME}_vs_baseline.txt"
  python -m pianovam_vision.score_midi --config "${DATA_CONFIG:-$CONFIG}" --split test \
      --pred "out/${NAME}_test" --baseline "$BASELINE_MIDI" \
      --csv "results/${NAME}_vs_baseline.csv" > "results/${NAME}_vs_baseline.txt" 2>&1 \
    && sed -n '/^mean over/,$p' "results/${NAME}_vs_baseline.txt" \
    || echo "FAILED: comparison -- see results/${NAME}_vs_baseline.txt"
fi
echo "=== $(date '+%Y-%m-%d %H:%M') ${NAME} done"
