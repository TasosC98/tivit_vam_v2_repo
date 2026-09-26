#!/usr/bin/env bash
# Phase 0 of EXPERIMENT_PLAN.md, unattended: the existing models re-scored under
# the paper protocols (first: the zero-shot PianoVAM -> PianoYT number), dataset
# checks, and the PianoYT contact sheets. Runs in the background, so closing
# PuTTY does not stop it.
#
#   bash scripts/phase0.sh
#   tail -f logs/phase0.log          # Ctrl+C stops watching, not the job
#
# If your old checkpoints live elsewhere, or were calibrated differently:
#   OLD_VAM=/path/best.pt OLD_YT=runs/<run>/best.pt bash scripts/phase0.sh
#   OLD_VAM_THR="decode.onset_threshold=0.6 decode.frame_threshold=0.5"   (same for OLD_YT_THR)
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

OLD_VAM="${OLD_VAM:-/home/achatzigiannis/tivit_vam_v2_repo/vision_transcription/runs/tiled_best_v2/best.pt}"
OLD_VAM_THR="${OLD_VAM_THR:-decode.onset_threshold=0.60 decode.frame_threshold=0.50}"
OLD_YT="${OLD_YT:-runs/pianoyt_tiled/best.pt}"
OLD_YT_THR="${OLD_YT_THR:-decode.onset_threshold=0.70 decode.frame_threshold=0.40}"
VAM_CONFIG="${VAM_CONFIG:-configs/vam_full.yaml}"      # dataset checks
YT_CONFIG="${YT_CONFIG:-configs/yt_full.yaml}"
YT_DATA="${YT_DATA:-configs/pianoyt.yaml}"             # PianoYT data for the old (360p) models

mkdir -p logs results
if [ -z "${_PHASE0_CHILD:-}" ]; then
  _PHASE0_CHILD=1 nohup bash "$SELF" "$@" > logs/phase0.log 2>&1 &
  echo "Phase 0 started in the background (pid $!); it takes a few hours."
  echo "  watch:   tail -f logs/phase0.log"
  echo "  results: results/phase0_*.txt, contact sheets in preview_keys_yt/"
  exit 0
fi

export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-2048}"
say() { echo; echo "=== $(date '+%Y-%m-%d %H:%M') $*"; }
# run <name> <command...>: output to results/phase0_<name>.txt, one-line status here
run() {
  local out="results/phase0_$1.txt"; shift
  if "$@" > "$out" 2>&1; then echo "   done -> $out"; else echo "   FAILED -> see $out"; fi
}

say "1/6 zero-shot: old PianoVAM model on the PianoYT test set (the key number)"
if [ -f "$OLD_VAM" ]; then
  run vam_on_yt python -m pianovam_vision.evaluate --config configs/tiled_best.yaml \
      --checkpoint "$OLD_VAM" --data_config "$YT_DATA" --split test \
      --csv results/phase0_vam_on_yt.csv $OLD_VAM_THR
else
  echo "   SKIPPED: no checkpoint at $OLD_VAM (set OLD_VAM=...)"
fi

say "2/6 old PianoVAM model on the PianoVAM test set (50 and 100 ms)"
if [ -f "$OLD_VAM" ]; then
  run vam_on_vam python -m pianovam_vision.evaluate --config configs/tiled_best.yaml \
      --checkpoint "$OLD_VAM" --split test --csv results/phase0_vam_on_vam.csv $OLD_VAM_THR
fi

say "3/6 old PianoYT model on the PianoYT test set (50 and 100 ms)"
if [ -f "$OLD_YT" ]; then
  run yt_on_yt python -m pianovam_vision.evaluate --config configs/pianoyt.yaml \
      --checkpoint "$OLD_YT" --split test --csv results/phase0_yt_on_yt.csv $OLD_YT_THR
else
  echo "   SKIPPED: no checkpoint at $OLD_YT (set OLD_YT=... to your PianoYT run)"
fi

say "4/6 PianoYT: per-video resolution / fps / duration / crop checks"
run probe_yt python -m pianovam_vision.check_data --config "$YT_CONFIG" --probe

say "5/6 PianoVAM: the same checks"
run probe_vam python -m pianovam_vision.check_data --config "$VAM_CONFIG" --probe

say "6/6 PianoYT contact sheets (keys struck in the last 100 ms, 10 videos per image)"
run sheets python -m pianovam_vision.draw_keyboard --config "$YT_CONFIG" \
    --busiest --onsets_only --sheet 10 --out_dir preview_keys_yt/

say "SUMMARY"
for n in vam_on_yt vam_on_vam yt_on_yt; do
  f="results/phase0_$n.txt"
  [ -f "$f" ] && { echo "--- $n"; grep -E "onset\+pitch|headline|evaluated" "$f" | sed 's/^/   /'; }
done
for n in probe_yt probe_vam; do
  f="results/phase0_$n.txt"
  [ -f "$f" ] && { echo "--- $n"; sed -n '/=== usable data per split ===/,$p' "$f" | head -40 | sed 's/^/   /'; }
done
echo
echo "Phase 0 finished. Send these files: results/phase0_vam_on_yt.txt results/phase0_vam_on_vam.txt"
echo "results/phase0_yt_on_yt.txt results/phase0_probe_yt.txt -- and look at preview_keys_yt/sheet_*.png"
