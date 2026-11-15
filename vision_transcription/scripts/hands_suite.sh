#!/usr/bin/env bash
# All KeyWise-hands experiments, one after another, in the background (closing PuTTY is fine).
#
#   bash scripts/preflight_hands.sh                       # first: is the server ready?
#   BASELINE_MIDI=/home/achatzigiannis/V2N/predicted_midi/pianovam_test \
#     bash scripts/hands_suite.sh [scripts/jobs_hands_suite.txt]
#
# Optional: HN_OVERRIDES="hand_notes.batch_windows=8"  extra overrides for every training run
# (e.g. when another job leaves little GPU memory); CUDA_VISIBLE_DEVICES=1 picks a GPU.
#
# Steps (each skips what is already done, so start it again after a crash or a reboot):
#   1. rules on all 11 manually labelled recordings (incl. At Your Fingertips' rule), CPU
#        -> results/hn/rules_allgt.txt, per-note scores in out/hn/rules_allgt/
#   2. every training job of the jobs file (hand_notes cv)          -> runs/hn/<run>/
#   3. robustness of loro_keyrel / session_keyrel / session_image (seed 1), test-time
#      corruptions of the landmarks                                 -> runs/hn/<run>/<fold>/pred_<test>/
#      (other runs: ROBUST_RUNS="loro_cornerjit loro_manual ...")
#   4. end to end on test: hands for KeyWise's notes (and V2N's with BASELINE_MIDI)
#        -> results/hn/e2e_<run>_s<seed>.txt, hand-separated MIDI in out/hn/e2e_<run>_s<seed>/
#   5. FUSION=1 only: per-note probabilities of the video model vam_hands (GPU + videos, ~2 h)
#        -> out/hn/fusion/ (valid + test), compared in results/hn/official_*.txt
#   6. every report: bash scripts/hands_reports.sh                  -> results/hn/*.txt
#
# Log: logs/hands_suite.log    Stop: kill $(cat logs/hands_suite.pid)
# Reports of what is finished so far, at any time: bash scripts/hands_reports.sh
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."
JOBS="${1:-scripts/jobs_hands_suite.txt}"
CONFIG="${HN_CONFIG:-configs/hand_notes.yaml}"
[ -f "$JOBS" ] || { echo "ERROR: no jobs file $JOBS"; exit 1; }
[ -f "$CONFIG" ] || { echo "ERROR: no config $CONFIG"; exit 1; }
mkdir -p logs results/hn out/hn runs/hn

if [ -z "${_SUITE_CHILD:-}" ]; then
  if [ -f logs/hands_suite.pid ] && kill -0 "$(cat logs/hands_suite.pid)" 2>/dev/null; then
    echo "ERROR: the suite is already running (pid $(cat logs/hands_suite.pid)); log: logs/hands_suite.log"
    exit 1
  fi
  while read -r name scheme seeds _; do
    case "$name" in ''|'#'*) continue;; esac
    case "$scheme" in official|loro|lopo|session) ;; *)
      echo "ERROR: job '$name': unknown scheme '$scheme'"; exit 1;; esac
    [[ "$seeds" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "ERROR: job '$name': seeds must look like 1,2,3"; exit 1; }
  done < "$JOBS"
  _SUITE_CHILD=1 nohup bash "$SELF" "$@" >> logs/hands_suite.log 2>&1 &
  echo $! > logs/hands_suite.pid
  echo "hands suite started in the background (pid $!)"
  echo "  log:     tail -f logs/hands_suite.log"
  echo "  one run: tail -f runs/hn/<run>/<fold>_s<seed>/train.log"
  echo "  stop:    kill \$(cat logs/hands_suite.pid)"
  exit 0
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
say() { echo "=== $(date '+%Y-%m-%d %H:%M') $*"; }
say "hands suite: ${JOBS} (config ${CONFIG})"

# ---- 1. rules on all manual labels
if ! grep -q '=== given notes: manual' results/hn/rules_allgt.txt 2>/dev/null; then
  say "rules on all 11 manually labelled recordings -> results/hn/rules_allgt.txt"
  python -m pianovam_vision.evaluate_hands --config "$CONFIG" \
      --split train,ext-train,valid,test --methods pitch,tip,roi,ayf,auto \
      --csv results/hn/rules_allgt.csv --save_probs out/hn/rules_allgt --save_probs_manual_only \
      > results/hn/rules_allgt.txt 2>&1 || say "FAILED: see results/hn/rules_allgt.txt"
fi

# ---- 2. trainings
while read -r name scheme seeds rest <&3; do
  case "$name" in ''|'#'*) continue;; esac
  read -ra ov <<< "${rest:-} ${HN_OVERRIDES:-}"
  say "${name}: ${scheme}, seeds ${seeds} ${rest:-}"
  if ! python -m pianovam_vision.hand_notes cv --config "$CONFIG" --scheme "$scheme" \
       --out "runs/hn/${name}" --seeds "$seeds" ${ov[@]+"${ov[@]}"}; then
    say "FAILED: ${name} (see runs/hn/${name}/*/train.log); going on with the next job"
  fi
done 3< "$JOBS"

# ---- 3. robustness
for run in ${ROBUST_RUNS:-loro_keyrel session_keyrel session_image}; do
  if [ -d "runs/hn/${run}" ]; then
    say "robustness: ${run} (seed 1)"
    python -m pianovam_vision.hand_notes robust --config "$CONFIG" --run_dir "runs/hn/${run}" --seeds 1 \
      || say "FAILED: robustness of ${run}"
  fi
done

# ---- 4. end to end
PRED=()
KW="${KEYWISE_MIDI:-out/E1s2_vam_full_test}"
[ -d "$KW" ] && PRED+=(--pred "KeyWise=${KW}") || say "no ${KW}: set KEYWISE_MIDI to KeyWise's test MIDI folder"
[ -n "${BASELINE_MIDI:-}" ] && PRED+=(--pred "V2N=${BASELINE_MIDI}")
if [ ${#PRED[@]} -gt 0 ]; then
  for run in official_keyrel official_none; do
    for fd in runs/hn/${run}/official_s*; do
      [ -f "${fd}/model.pt" ] || continue
      s="${fd##*_s}"
      OUTF="results/hn/e2e_${run}_s${s}"
      grep -q 'hand-separated MIDI' "${OUTF}.txt" 2>/dev/null && continue
      say "end to end: ${run} seed ${s} -> ${OUTF}.txt"
      python -m pianovam_vision.hand_notes e2e --config "$CONFIG" --model "${fd}/model.pt" --split test \
          "${PRED[@]}" --out "out/hn/e2e_${run}_s${s}" --csv "${OUTF}.csv" > "${OUTF}.txt" 2>&1 \
        || say "FAILED: see ${OUTF}.txt"
    done
  done
fi

# ---- 5. optional: the video model's per-note probabilities (same reports)
if [ "${FUSION:-0}" = "1" ] && [ -f runs/vam_hands/best.pt ]; then
  for SPLIT in valid test; do
    grep -q '=== given notes: manual' "results/hn/fusion_${SPLIT}.txt" 2>/dev/null && continue
    say "video model vam_hands, ${SPLIT}: per-note probabilities -> out/hn/fusion/"
    LINE="$(grep -E '^[[:space:]]*decode\.onset_threshold=' results/E1s2_vam_full_calibrate.txt | tail -n 1 | sed 's/^[[:space:]]*//')"
    OT="${LINE#decode.onset_threshold=}"; OT="${OT%% *}"
    FT="${LINE#*decode.frame_threshold=}"; FT="${FT%% *}"
    python -m pianovam_vision.evaluate_hands --config "$CONFIG" --checkpoint runs/vam_hands/best.pt \
        --split "$SPLIT" --methods model,roi,ayf,pitch,tip,auto --save_probs out/hn/fusion \
        "decode.onset_threshold=${OT}" "decode.frame_threshold=${FT}" \
        > "results/hn/fusion_${SPLIT}.txt" 2>&1 || say "FAILED: see results/hn/fusion_${SPLIT}.txt"
  done
fi

# ---- 6. reports
say "reports"
bash scripts/hands_reports.sh
say "hands suite done"
