#!/usr/bin/env bash
# One screen with everything: trainings, evaluations, queued jobs, GPU, disk.
#
#   bash scripts/report.sh
#   bash scripts/report.sh | tee results/report.txt     # also keep a copy to download
set -uo pipefail
cd "$(dirname "$0")/.."
shopt -s nullglob

echo "===== trainings"
bash scripts/status.sh

echo; echo "===== evaluations"
for f in logs/E*.log logs/A[0-9]*.log logs/P0*.log; do
  n="$(basename "$f" .log)"
  state="finished"
  if [ -f "logs/${n}.pid" ] && kill -0 "$(cat "logs/${n}.pid")" 2>/dev/null; then
    state="RUNNING"
  fi
  lines="$(grep -aE "calibrating|reusing|scoring|headline|FAILED|done|waiting" "$f" | tail -n 4)"
  if [ "$state" = "RUNNING" ] && printf '%s\n' "$lines" | tail -n 1 | grep -q "waiting:"; then
    state="WAITING for another evaluation"
  fi
  echo "--- ${n}  [${state}]"
  [ -n "$lines" ] && printf '%s\n' "$lines" | sed 's/^/    /'
  # Progress of the phase the run is in now. (A calibration file left by an
  # earlier run of the same name says nothing about a run that only scores test.)
  if [ "$state" = "RUNNING" ]; then
    calib="results/${n}_calibrate.txt"; test_out="results/${n}_test.txt"
    case "$(printf '%s\n' "$lines" | tail -n 1)" in
      *calibrating*)
        [ -f "$calib" ] && echo "    calibration progress: $(grep -acE '(predicted|loaded) rolls' "$calib")" \
             "videos predicted, $(grep -acE 'onset_f1 [0-9]|skipped \(cannot' "$calib") thresholds tried" ;;
      *scoring*)
        if [ -f "$test_out" ]; then
          total="$(grep -aoE '\([0-9]+ recordings\)' "$test_out" | head -n 1 | tr -dc '0-9')"
          echo "    test progress: $(grep -acE '^[^ ]+: (onset_f1@50=|\[skipped\])' "$test_out")" \
               "of ${total:-?} videos scored"
        fi ;;
    esac
  fi
done

echo; echo "===== queued (after.sh)"
for f in logs/after_*.log; do
  echo "--- ${f}"
  tail -n 2 "$f" | sed 's/^/    /'
done

echo; echo "===== GPU"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null \
  || echo "    (nvidia-smi not available)"
echo; echo "===== disk"
df -h /raid_storage 2>/dev/null | tail -n 1 || true
