#!/usr/bin/env bash
# Every KeyWise-hands report from what has finished so far (safe while the suite runs:
# unfinished runs are left out or compared on the recordings they have done).
#
#   bash scripts/hands_reports.sh          -> results/hn/*.txt (+ per-note CSV, curves)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/hn
R=runs/hn

rep() {   # rep <output name> [hand_report args...]
  local out="results/hn/$1"; shift
  if python -m pianovam_vision.hand_report "$@" --out "${out}.txt" > "${out}.log" 2>&1; then
    echo "--- ${out}.txt"; sed -n '1,/^  differences/p' "${out}.txt" | head -n 40
  else
    echo "--- ${out}: not yet ($(grep -v '^  (run' "${out}.log" | tail -n 1))"
  fi
}

# Main results: cross-validation over all 1,800 manual notes
rep loro_manual --run keyrel=$R/loro_keyrel --run midi_only=$R/loro_none --run image=$R/loro_image \
    --run strip=$R/loro_strip --run hand_only=$R/loro_handsup --main keyrel:seq \
    --csv results/hn/loro_notes.csv --curves results/hn/curves_loro
rep lopo_manual --run keyrel=$R/lopo_keyrel --run midi_only=$R/lopo_none --run image=$R/lopo_image \
    --main keyrel:seq --csv results/hn/lopo_notes.csv
rep session_manual --run keyrel=$R/session_keyrel --run midi_only=$R/session_none --run image=$R/session_image \
    --run strip=$R/session_strip --run corner_jitter=$R/session_cornerjit --main keyrel:seq
# Official split: 450 unseen manual notes, and every automatic label of valid + test
OFF=(--run keyrel=$R/official_keyrel --run midi_only=$R/official_none --run image=$R/official_image
     --run strip=$R/official_strip --run no_seq=$R/official_noseq --run no_key_ctx=$R/official_nokeyctx
     --run no_time_ctx=$R/official_notimectx --run hand_only=$R/official_handsup
     --run corner_jitter=$R/official_cornerjit)
[ -d out/hn/fusion ] && OFF+=(--run fusion_video=out/hn/fusion)
rep official_manual "${OFF[@]}" --main keyrel:seq --curves results/hn/curves_official
rep official_automatic "${OFF[@]}" --labels automatic --main keyrel:seq --decoders seq,frame --n_boot 500
# Rules alone on all 11 recordings (no training involved)
[ -d out/hn/rules_allgt ] && rep rules_allgt_manual --run rules=out/hn/rules_allgt --main roi
# Robustness: test-time corruptions (seed 1)
for t in corner0.005 corner0.01 corner0.02 noise0.1 noise0.25 swap0.05 swap0.2 drop0.2 drop0.5; do
  rep "robust_loro_${t}" --run keyrel=$R/loro_keyrel --pred_dir "pred_${t}" --main keyrel:seq --n_boot 500
  rep "robust_session_${t}" --run keyrel=$R/session_keyrel --run image=$R/session_image \
      --pred_dir "pred_${t}" --main keyrel:seq --n_boot 500
done
# Round 2 (scripts/jobs_hands_round2.txt): corner jitter and manual labels
if ls -d $R/loro_cornerjit $R/loro_manual >/dev/null 2>&1; then
  rep round2_loro_manual --run keyrel=$R/loro_keyrel --run corner_jitter=$R/loro_cornerjit \
      --run manual=$R/loro_manual --run cj_manual=$R/loro_cj_manual --run cj0025=$R/loro_cornerjit0025 \
      --run cj01=$R/loro_cornerjit01 --run manual25=$R/loro_manual25 --main keyrel:seq \
      --csv results/hn/round2_loro_notes.csv
  rep round2_lopo_manual --run keyrel=$R/lopo_keyrel --run corner_jitter=$R/lopo_cornerjit \
      --run manual=$R/lopo_manual --main keyrel:seq
  for t in corner0.005 corner0.01 corner0.02 noise0.1 noise0.25 swap0.05 swap0.2 drop0.2 drop0.5; do
    rep "robust2_loro_${t}" --run keyrel=$R/loro_keyrel --run corner_jitter=$R/loro_cornerjit \
        --run manual=$R/loro_manual --run cj_manual=$R/loro_cj_manual --run cj0025=$R/loro_cornerjit0025 \
        --run cj01=$R/loro_cornerjit01 --pred_dir "pred_${t}" --main keyrel:seq --n_boot 500
  done
fi
echo
echo "end to end: results/hn/e2e_*.txt"
for f in results/hn/e2e_*.txt; do
  [ -f "$f" ] || continue
  echo "--- $f"; sed -n '/=== end-to-end/,/error split/p' "$f" | head -n 30
done
