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
for f in logs/E*.log logs/P0*.log; do
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
  calib="results/${n}_calibrate.txt"
  if [ "$state" = "RUNNING" ] && [ -f "$calib" ]; then
    echo "    calibration progress: $(grep -ac 'predicted rolls' "$calib") videos predicted," \
         "$(grep -acE 'onset_f1 [0-9]|skipped \(cannot' "$calib") thresholds tried"
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
