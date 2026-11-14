#!/usr/bin/env bash
# Is this server ready for the KeyWise-hands experiments (scripts/hands_suite.sh)?
# Read-only checks + a 1-2 minute GPU speed test. Run it in the foreground:
#
#   cd ~/<repo>/vision_transcription && bash scripts/preflight_hands.sh
#
# Ends with READY or NOT READY and the reasons.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
problems=()
warn=()

echo "================ 1. GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  problems+=("no nvidia-smi: no GPU driver on this machine")
else
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  echo "--- processes using the GPU (user, pid, running for, command):"
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
  if [ -z "$pids" ]; then
    echo "    none"
  else
    for p in $pids; do ps -o user=,pid=,etime=,args= -p "$p" 2>/dev/null | cut -c1-160; done
  fi
  echo "--- your own jobs that may still run:"
  for f in logs/hands_suite.pid logs/queue_*.pid runs/*/run.pid; do
    [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && echo "    RUNNING: $f (pid $(cat "$f"))"
  done
  echo "    (end of list)"
fi
python - <<'EOF'
import sys, torch
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit(1)
free, total = torch.cuda.mem_get_info()
print(f"GPU {torch.cuda.current_device()}: {torch.cuda.get_device_name()} -- free {free / 2**30:.1f} of {total / 2**30:.1f} GB")
if free < 6 * 2**30:
    print("WARNING: under 6 GB free. Wait for the other jobs, or train with hand_notes.batch_windows=8")
    sys.exit(2)
EOF
rc=$?
[ $rc -eq 1 ] && problems+=("PyTorch cannot use the GPU (see section 1)")
[ $rc -eq 2 ] && warn+=("little free GPU memory (see section 1)")

echo
echo "================ 2. code"
echo "branch: $(git branch --show-current)   commit: $(git log --oneline -1)"
if git status --porcelain -- pianovam_vision configs scripts tests | grep -q .; then
  git status --short -- pianovam_vision configs scripts tests
  warn+=("uncommitted changes in the code (fine if intended)")
fi
for f in pianovam_vision/hand_notes.py pianovam_vision/hand_report.py ${HN_CONFIG:-configs/hand_notes.yaml} \
         scripts/hands_suite.sh scripts/jobs_hands_suite.txt; do
  [ -f "$f" ] || problems+=("missing $f: pull the KeyWise-hands branch")
done
python -c "import matplotlib" 2>/dev/null || warn+=("no matplotlib: reports will have no PNG curves (pip install matplotlib)")
python -c "import scipy" 2>/dev/null || warn+=("no scipy: chord matching unavailable (not used by the suite)")
python -m pytest tests/test_hand_notes.py tests/test_hands.py tests/test_imports.py -q > logs/preflight_tests.log 2>&1
rc=$?
tail -n 3 logs/preflight_tests.log
[ $rc -eq 0 ] || problems+=("unit tests fail (logs/preflight_tests.log)")

echo
echo "================ 3. data"
python - <<'EOF' || problems+=("data check failed (see section 3)")
import os, sys
from pathlib import Path
from pianovam_vision.config import load_config
from pianovam_vision import hands as H
from pianovam_vision.metadata import recordings_from_cfg
cfg = load_config(os.environ.get("HN_CONFIG", "configs/hand_notes.yaml"))
d = cfg["data"]
print(f"profile: {cfg.get('_active_profile')} | data.root {d['root']} | skeleton_cache {d.get('skeleton_cache') or '(none)'}")
recs = [r for r in recordings_from_cfg(cfg) if r.split in ("train", "ext-train", "valid", "test")
        and r.record_time not in set(d.get("exclude_records") or [])]
fing = [r for r in recs if H.fingering_path(cfg, r.record_time).exists()]
gt = [r for r in recs if H.fingering_path(cfg, r.record_time, gt=True).exists()]
cache = [r for r in recs if (H.skeleton_cache_path(cfg, r.record_time) or Path("/nonexistent")).exists()]
js = [r for r in recs if H.skeleton_json_path(cfg, r.record_time).exists()]
print(f"recordings (train/ext-train/valid/test): {len(recs)} | Fingering {len(fing)} | Fingering_GT {len(gt)} "
      f"| skeleton cache {len(cache)} | skeleton JSON {len(js)}")
bad = []
if len(gt) != 11:
    bad.append(f"expected 11 recordings with manual labels, found {len(gt)} (scripts/download_hand_data.py)")
if len(fing) < 98:
    bad.append(f"only {len(fing)} recordings with automatic labels (expected 98)")
if len(cache) < len(fing):
    print(f"WARNING: {len(fing) - len(cache)} recordings have no skeleton cache: the first run parses their JSON "
          f"(slow). Build it: python -m pianovam_vision.check_hands --config configs/vam_hands.yaml --skeleton --workers 4")
off = d.get("skeleton_offsets")
if not off or not Path(off).exists():
    bad.append(f"no landmark offsets file {off!r}")
if bad:
    print("PROBLEM: " + "; ".join(bad))
    sys.exit(1)
# one cache file read end to end, with the settings of this config
r = gt[0]
tr = H.load_skeleton(cfg, r, write_cache=False)
print(f"{r.record_time}: {len(tr)} frames of landmarks, both hands seen in {100 * tr.present.all(1).mean():.0f} % of them")
EOF
echo "--- folds:"
python -m pianovam_vision.hand_notes folds --config ${HN_CONFIG:-configs/hand_notes.yaml} --scheme official,loro,lopo,session \
  || problems+=("cannot build the folds")
[ -d out/E1s2_vam_full_test ] || warn+=("no out/E1s2_vam_full_test (KeyWise's test MIDI): the end-to-end step is skipped (set KEYWISE_MIDI)")

echo
echo "================ 4. GPU speed test (60 training steps on 8 recordings)"
if [ ${#problems[@]} -eq 0 ]; then
  rm -rf runs/hn/_preflight
  python -m pianovam_vision.hand_notes train --config ${HN_CONFIG:-configs/hand_notes.yaml} --out runs/hn/_preflight \
      --fold loro:2024-02-17_21-44-37 hand_notes.steps=60 hand_notes.log_every=20 hand_notes.limit_recordings=8 \
      2>&1 | grep -E "model:|step |trained in|manual notes|Error|error" | cut -c1-200
  sps="$(grep -h 'trained in' runs/hn/_preflight/*/train.log 2>/dev/null | sed -E 's/.*\(([0-9.]+) s\/step\).*/\1/')"
  if [ -n "$sps" ]; then
    python - "$sps" <<'EOF'
import sys
s = float(sys.argv[1])
run = s * 3000 / 60 + 1.0          # + loading / predicting
folds = {"official": 1, "loro": 11, "lopo": 7, "session": 3}
total = 0
for line in open("scripts/jobs_hands_suite.txt"):
    p = line.split()
    if not p or p[0].startswith("#"):
        continue
    total += folds[p[1]] * len(p[2].split(","))
print(f"{s:.3f} s/step -> about {run:.0f} min per training run; the suite has {total} runs: about "
      f"{total * run / 60:.0f} h of GPU time (+ ~1 h for rules, robustness, end to end)")
EOF
    rm -rf runs/hn/_preflight
  else
    problems+=("the GPU speed test failed (see section 4)")
  fi
else
  echo "skipped (fix the problems first)"
fi

echo
echo "================ VERDICT"
for w in "${warn[@]+"${warn[@]}"}"; do echo "  warning: $w"; done
if [ ${#problems[@]} -eq 0 ]; then
  echo "READY. Start everything with:"
  echo "  BASELINE_MIDI=/home/achatzigiannis/V2N/predicted_midi/pianovam_test bash scripts/hands_suite.sh"
else
  echo "NOT READY:"
  for p in "${problems[@]}"; do echo "  - $p"; done
  exit 1
fi
