#!/usr/bin/env bash
# Run one complete training experiment and keep its results in runs/<name>/.
#
# Usage:
#   bash scripts/run_experiment.sh <name> <arch> [extra config overrides...]
#
# Examples:
#   # GPU server (dib) — full-size experiment:
#   bash scripts/run_experiment.sh tiled_v1 tiled
#   bash scripts/run_experiment.sh strip_v1 strip
#
#   # CPU server (dit) — lighter so epochs finish in reasonable time:
#   bash scripts/run_experiment.sh tiled_cpu tiled train.batch_size=2 train.max_frames_per_record=300
#
#   # any extra override just gets appended, e.g. a longer run:
#   bash scripts/run_experiment.sh tiled_long tiled train.epochs=40
#
# The active server profile (paths + cpu/gpu) is auto-selected by hostname,
# so the SAME command works on both servers. Results, log, and checkpoints
# all land under runs/<name>/. Re-using a name is refused so you never
# overwrite a previous experiment.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo's vision_transcription/ dir

NAME="${1:?usage: run_experiment.sh <name> <arch=strip|tiled> [overrides...]}"
ARCH="${2:?arch must be 'strip' or 'tiled'}"
shift 2

# Raise decord's EOF retry limit (some PianoVAM mp4s seek slowly near the end).
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-40960}"
# Cap CPU thread fan-out (polite on shared boxes; harmless on the GPU server).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

OUT="runs/${NAME}"
if [ -e "${OUT}" ]; then
  echo "ERROR: ${OUT} already exists — pick a new <name> or 'rm -rf ${OUT}' first." >&2
  exit 1
fi
mkdir -p "${OUT}"

# Stable defaults: num_workers=0 avoids the flaky decord-in-subprocess crashes.
# Memory-safe caps + a frame cap keep epochs tractable. Override any of these
# by passing e.g. train.epochs=40 on the command line.
echo "Launching '${NAME}' (arch=${ARCH}) -> ${OUT}"
nohup python -m pianovam_vision.train \
    --config configs/default.yaml \
    model.arch="${ARCH}" \
    train.num_workers=0 \
    train.max_open_readers=2 \
    train.max_cached_targets=8 \
    keyboard.decode_height=360 \
    train.max_frames_per_record=600 \
    train.batch_size=8 \
    train.epochs=20 \
    train.out_dir="${OUT}" \
    "$@" \
    > "${OUT}/train.log" 2>&1 &

PID=$!
echo "${PID}" > "${OUT}/run.pid"
echo "started PID ${PID}"
echo "  watch:    tail -f ${OUT}/train.log"
echo "  results:  grep valid ${OUT}/train.log"
echo "  stop:     kill \$(cat ${OUT}/run.pid)"
