#!/usr/bin/env bash
# Run (or resume) one complete training experiment, keeping results in runs/<name>/.
#
# Usage:
#   bash scripts/run_experiment.sh <name> <arch> [extra config overrides...]
#
# Env vars:
#   CONFIG=configs/tiled_best.yaml   # which config file to use (default: default.yaml)
#
# Examples:
#   # the pinned best experiment (config carries all the winning settings):
#   CONFIG=configs/tiled_best.yaml bash scripts/run_experiment.sh tiled_best tiled
#
#   # a quick variant on the default config:
#   bash scripts/run_experiment.sh tiled_v3 tiled keyboard.warp_width=1408
#
# If runs/<name>/last.pt already exists, the run RESUMES from it (so a crash
# costs you nothing). The active server profile (paths + cpu/gpu) is auto-
# selected by hostname, so the same command works on both servers.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?usage: run_experiment.sh <name> <arch=strip|tiled> [overrides...]}"
ARCH="${2:?arch must be 'strip' or 'tiled'}"
shift 2

CONFIG="${CONFIG:-configs/default.yaml}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-40960}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

OUT="runs/${NAME}"
mkdir -p "${OUT}"

# Refuse to launch a duplicate if this experiment is already running.
if [ -f "${OUT}/run.pid" ] && kill -0 "$(cat "${OUT}/run.pid")" 2>/dev/null; then
  echo "ERROR: '${NAME}' is already running (pid $(cat "${OUT}/run.pid")). " \
       "Stop it first:  kill \$(cat ${OUT}/run.pid)" >&2
  exit 1
fi

RESUME=""
if [ -f "${OUT}/last.pt" ]; then
  RESUME="--resume auto"
  echo "Resuming '${NAME}' from ${OUT}/last.pt"
else
  echo "Launching '${NAME}' (arch=${ARCH}, config=${CONFIG}) -> ${OUT}"
fi

# num_workers=0 is the safe default; pass train.num_workers=4 to go faster on a
# healthy box. Resume + the dataset's skip-bad-clip logic make long runs robust.
nohup python -m pianovam_vision.train \
    --config "${CONFIG}" ${RESUME} \
    model.arch="${ARCH}" \
    train.out_dir="${OUT}" \
    "$@" \
    >> "${OUT}/train.log" 2>&1 &

PID=$!
echo "${PID}" > "${OUT}/run.pid"
echo "started PID ${PID}"
echo "  watch:    tail -f ${OUT}/train.log"
echo "  status:   bash scripts/status.sh"
echo "  stop:     kill \$(cat ${OUT}/run.pid)"
