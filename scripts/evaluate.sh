#!/usr/bin/env bash
# Evaluate one trained model as in the paper: calibrate the decoding thresholds on the
# validation split, then score the test split with exactly those thresholds.
#
#   bash scripts/evaluate.sh <name> <config> <checkpoint> <target>
#     <name>    label for the outputs, e.g. pianovam_s1
#     <target>  full_f1   PianoVAM (thresholds chosen on onset + offset F1)
#               onset_f1  PianoYT  (its offsets include the sustain pedal)
#
# Optional environment variables:
#   DATA_CONFIG=configs/pianoyt.yaml   dataset to calibrate and test on (default: <config>);
#                                      a PianoVAM model on PianoYT gives the zero-shot result
#   BASELINE_MIDI=/path/to/V2N/predicted_midi/pianovam_test
#                                      also score V2N's released MIDI with the same code
#                                      (results/<name>_vs_baseline.csv, read by paper/*.py)
#
# Outputs: results/<name>_calibrate.txt, results/<name>_test.txt and .csv (per recording),
#          out/<name>_test/*.mid, and with BASELINE_MIDI results/<name>_vs_baseline.txt and .csv.
# The model's probability rolls are kept in rolls/<name>/, so running it again is quick.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -ne 4 ]; then
  sed -n '2,19p' "scripts/evaluate.sh"
  exit 1
fi
NAME="$1"; CONFIG="$2"; CKPT="$3"; TARGET="$4"
DATA="${DATA_CONFIG:-$CONFIG}"
case "$TARGET" in
  onset_f1|full_f1) ;;
  *) echo "ERROR: <target> must be onset_f1 or full_f1" >&2; exit 1 ;;
esac
mkdir -p results out

echo "--- calibrating the thresholds on valid (target ${TARGET}) -> results/${NAME}_calibrate.txt"
python -m pianovam_vision.calibrate --config "$CONFIG" --checkpoint "$CKPT" \
    --data_config "$DATA" --split valid --target "$TARGET" \
    --rolls_dir "rolls/${NAME}/valid" > "results/${NAME}_calibrate.txt"
DECODE="$(grep -E '^[[:space:]]*decode\.onset_threshold=' "results/${NAME}_calibrate.txt" \
          | tail -n 1 | sed 's/^[[:space:]]*//')"
echo "    ${DECODE}"

echo "--- scoring test -> results/${NAME}_test.txt"
# DECODE holds two key=value overrides: split it into words on purpose.
# shellcheck disable=SC2086
python -m pianovam_vision.evaluate --config "$CONFIG" --checkpoint "$CKPT" \
    --data_config "$DATA" --split test --csv "results/${NAME}_test.csv" \
    --save_midi "out/${NAME}_test" --rolls_dir "rolls/${NAME}/test" \
    $DECODE > "results/${NAME}_test.txt"
sed -n '/=== mean over recordings/,$p' "results/${NAME}_test.txt"

if [ -n "${BASELINE_MIDI:-}" ]; then
  echo "--- scoring out/${NAME}_test and ${BASELINE_MIDI} with the same code"
  python -m pianovam_vision.score_midi --config "$DATA" --split test \
      --pred "out/${NAME}_test" --baseline "$BASELINE_MIDI" \
      --csv "results/${NAME}_vs_baseline.csv" > "results/${NAME}_vs_baseline.txt"
  sed -n '/^mean over/,$p' "results/${NAME}_vs_baseline.txt"
fi
