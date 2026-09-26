#!/usr/bin/env bash
# Phase 1 of EXPERIMENT_PLAN.md: build the warped-strip caches that the paper runs
# train from (PianoVAM, then PianoYT), in the background. Re-running resumes:
# finished videos are skipped.
#
#   bash scripts/build_caches.sh           # workers = half the cores (max 8)
#   bash scripts/build_caches.sh 6         # explicit number of worker processes
#   tail -f logs/build_caches.log
#
# It refuses to start when the disk looks too small for what is left to build
# (~80 GB per dataset at 720p): QUALITY=85 makes the cache ~15% smaller,
# FORCE=1 skips the check. CONFIGS="configs/vam_full_360.yaml" builds another one.
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

CONFIGS="${CONFIGS:-configs/vam_full.yaml configs/yt_full.yaml}"
QUALITY="${QUALITY:-90}"
GB_PER_CACHE="${GB_PER_CACHE:-80}"
WORKERS="${1:-}"
if [ -z "$WORKERS" ]; then
  WORKERS=$(( $(nproc) / 2 ))
  [ "$WORKERS" -gt 8 ] && WORKERS=8
  [ "$WORKERS" -lt 1 ] && WORKERS=1
fi
mkdir -p logs

if [ -z "${_CACHE_CHILD:-}" ]; then
  # Disk check here, in the foreground, so the answer is on screen.
  need=0; dirs=""
  for c in $CONFIGS; do
    d="$(python -c "from pianovam_vision.config import load_config as L; print(L('$c')['data'].get('strip_cache') or '')")"
    if [ -z "$d" ]; then echo "ERROR: $c has no data.strip_cache on this machine"; exit 1; fi
    mkdir -p "$d" || { echo "ERROR: cannot create $d"; exit 1; }
    have=$(du -sk "$d" | awk '{print int($1 / 1048576)}')
    left=$(( GB_PER_CACHE - have )); [ "$left" -lt 0 ] && left=0
    need=$(( need + left )); dirs="$dirs $d"
    echo "cache for $c: $d (already ${have} GB)"
  done
  free=$(df -Pk $dirs | awk 'NR==2 {print int($4 / 1048576)}')
  echo "estimated space still needed: ~${need} GB; free on that disk: ${free} GB"
  if [ "$free" -lt "$need" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: not enough free space. Free some, or use QUALITY=85, or FORCE=1."
    exit 1
  fi
  _CACHE_CHILD=1 nohup bash "$SELF" "$WORKERS" > logs/build_caches.log 2>&1 &
  echo "cache build started in the background (pid $!, $WORKERS workers)."
  echo "  watch: tail -f logs/build_caches.log"
  exit 0
fi

export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-2048}"
for c in $CONFIGS; do
  echo; echo "=== $(date '+%Y-%m-%d %H:%M') building cache for $c ($WORKERS workers, quality $QUALITY)"
  python -m pianovam_vision.build_cache --config "$c" --workers "$WORKERS" --quality "$QUALITY"
done
echo; echo "=== $(date '+%Y-%m-%d %H:%M') all caches done"
for c in $CONFIGS; do
  d="$(python -c "from pianovam_vision.config import load_config as L; print(L('$c')['data']['strip_cache'])")"
  du -sh "$d"
done
