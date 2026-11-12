#!/usr/bin/env bash
# Score a KeyWise-hands run on PianoVAM, in the background (see HANDS.md).
#
#   bash scripts/evaluate_hands.sh <name> <config> <checkpoint>
#     e.g. bash scripts/evaluate_hands.sh vam_hands configs/vam_hands.yaml runs/vam_hands/best.pt
#
# Optional environment variables:
#   THRESHOLDS_FROM=results/E1s2_vam_full_calibrate.txt   decode thresholds of the frozen
#                         transcription network (default: this file, E1 run 2)
#   THRESHOLDS="0.90 0.70"                               or give them directly (onset frame)
#   BASELINE_MIDI=~/V2N/predicted_midi/pianovam_test     V2N's notes: hands by our model and the rules
#   KEYWISE_MIDI=out/E1s2_vam_full_test                  KeyWise's saved notes (default if present)
#
# Outputs: logs/<name>_hands.log (watch this);
#   results/hands_rules_allgt.txt        rules vs the dataset labels on all 1,800 manual notes (once)
#   results/<name>_hands_{valid,test}.txt + .csv          our model + every baseline, both settings
#   results/<name>_hands_test_chords.txt                  the same with chord matching
#   out/<name>_hands_demo/               transcription with hands of test 2024-02-17_21-44-37
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."
if [ $# -ne 3 ]; then
  sed -n '2,20p' "$SELF"; exit 1
fi
NAME="$1"; CONFIG="$2"; CKPT="$3"
[ -f "$CONFIG" ] || { echo "ERROR: no config $CONFIG"; exit 1; }
[ -f "$CKPT" ] || { echo "ERROR: no checkpoint $CKPT"; exit 1; }
mkdir -p logs results out
PIDFILE="logs/${NAME}_hands.pid"
if [ -z "${_EVAL_CHILD:-}" ]; then
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "ERROR: '${NAME}' hands evaluation is already running (pid $(cat "$PIDFILE"))"; exit 1
  fi
  _EVAL_CHILD=1 nohup bash "$SELF" "$@" > "logs/${NAME}_hands.log" 2>&1 &
  echo $! > "$PIDFILE"
  echo "hands evaluation '${NAME}' started in the background (pid $!)."
  echo "  watch: tail -f logs/${NAME}_hands.log"
  exit 0
fi
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-2048}"
export PYTHONUNBUFFERED=1
echo "=== $(date '+%Y-%m-%d %H:%M') ${NAME}: ${CKPT}"
# One GPU evaluation at a time (shared lock with evaluate_run.sh).
if command -v flock >/dev/null 2>&1; then
  exec 9> logs/.evaluate.lock
  if ! flock -n 9; then
    echo "--- waiting: another evaluation is using the GPU"; flock 9
  fi
fi

if [ -n "${THRESHOLDS:-}" ]; then
  read -r OT FT <<< "$THRESHOLDS"
else
  SRC="${THRESHOLDS_FROM:-results/E1s2_vam_full_calibrate.txt}"
  [ -f "$SRC" ] || { echo "FAILED: no $SRC (set THRESHOLDS_FROM or THRESHOLDS)"; exit 1; }
  LINE="$(grep -E '^[[:space:]]*decode\.onset_threshold=' "$SRC" | tail -n 1 | sed 's/^[[:space:]]*//')"
  OT="${LINE#decode.onset_threshold=}"; OT="${OT%% *}"
  FT="${LINE#*decode.frame_threshold=}"; FT="${FT%% *}"
  [ -n "$OT" ] && [ -n "$FT" ] || { echo "FAILED: no thresholds in $SRC"; exit 1; }
  echo "--- thresholds from ${SRC}"
fi
DECODE="decode.onset_threshold=${OT} decode.frame_threshold=${FT}"
echo "    ${DECODE}"

PRED=()
KW="${KEYWISE_MIDI:-out/E1s2_vam_full_test}"
[ -d "$KW" ] && PRED+=(--pred "KeyWise-MIDI=${KW}")
[ -n "${BASELINE_MIDI:-}" ] && PRED+=(--pred "V2N=${BASELINE_MIDI}")

if [ ! -f results/hands_rules_allgt.txt ]; then
  echo "--- rules vs dataset labels on all 11 manually labelled recordings -> results/hands_rules_allgt.txt"
  python -m pianovam_vision.evaluate_hands --config "$CONFIG" --split train,ext-train,valid,test \
      --methods pitch,tip,roi,auto --csv results/hands_rules_allgt.csv \
      > results/hands_rules_allgt.txt 2>&1 || echo "FAILED (see the file)"
  sed -n '/=== given notes: manual/,/=== given notes: automatic/p' results/hands_rules_allgt.txt
fi
for SPLIT in valid test; do
  OUTF="results/${NAME}_hands_${SPLIT}"
  echo "--- ${SPLIT} -> ${OUTF}.txt"
  P=()
  [ "$SPLIT" = test ] && P=(${PRED[@]+"${PRED[@]}"})
  # DECODE is a list of key=value overrides: split it into words on purpose.
  # shellcheck disable=SC2086
  if ! python -m pianovam_vision.evaluate_hands --config "$CONFIG" --checkpoint "$CKPT" \
       --split "$SPLIT" ${P[@]+"${P[@]}"} --csv "${OUTF}.csv" --save_dir "out/${NAME}_hands_${SPLIT}" \
       $DECODE > "${OUTF}.txt" 2>&1; then
    echo "FAILED: see ${OUTF}.txt"; exit 1
  fi
  sed -n '/=== given notes/,$p' "${OUTF}.txt"
done
echo "--- test with chord matching -> results/${NAME}_hands_test_chords.txt"
# shellcheck disable=SC2086
python -m pianovam_vision.evaluate_hands --config "$CONFIG" --checkpoint "$CKPT" --split test \
    --chord_matching ${PRED[@]+"${PRED[@]}"} $DECODE > "results/${NAME}_hands_test_chords.txt" 2>&1 \
    || echo "FAILED: see results/${NAME}_hands_test_chords.txt"
echo "--- demo -> out/${NAME}_hands_demo/"
# shellcheck disable=SC2086
python -m pianovam_vision.transcribe_hands --config "$CONFIG" --checkpoint "$CKPT" \
    --record_time 2024-02-17_21-44-37 --out_dir "out/${NAME}_hands_demo" --frames 6 $DECODE \
    | tail -n 32
echo "=== $(date '+%Y-%m-%d %H:%M') done"
