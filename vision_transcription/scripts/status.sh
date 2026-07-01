#!/usr/bin/env bash
# Print a one-line status for every experiment under runs/: whether it is still
# running, its latest epoch/progress, and its most recent validation F1.
#
#   bash scripts/status.sh              # snapshot
#   watch -n 10 bash scripts/status.sh  # live, refreshing every 10s
set -uo pipefail
cd "$(dirname "$0")/.."
shopt -s nullglob

found=0
for d in runs/*/; do
  log="${d}train.log"
  [ -f "${log}" ] || continue
  found=1
  name="$(basename "${d}")"

  # Running or not (via the pid the launcher recorded).
  status="stopped/done"
  if [ -f "${d}run.pid" ]; then
    pid="$(cat "${d}run.pid")"
    if kill -0 "${pid}" 2>/dev/null; then status="RUNNING (pid ${pid})"; fi
  fi

  # tqdm writes progress with carriage returns; turn them into lines and take
  # the most recent epoch/progress fragment.
  prog="$(tr '\r' '\n' < "${log}" | grep -E 'epoch [0-9]+:' | tail -n 1)"
  # Latest finished-epoch validation line, if any.
  valid="$(grep -E '\[epoch [0-9]+\] valid' "${log}" | tail -n 1)"

  echo "=== ${name}  [${status}] ==="
  [ -n "${prog}" ]  && echo "    ${prog}"
  [ -n "${valid}" ] && echo "    ${valid}"
done

[ "${found}" -eq 0 ] && echo "no experiments found under runs/"
