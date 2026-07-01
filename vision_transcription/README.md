# PianoVAM — Video-only Piano Transcription

Recognise pressed piano keys (**onset**, **key-release offset**, **pitch**) directly
from the keyboard **video** and export a **MIDI** file. **Audio is never used** —
the model only looks at the keys.

This implements a visual *Onsets-and-Frames* network (the acoustic spectrogram is
replaced by a perspective-rectified image of the keyboard), trained on the
**PianoVAM v1.0** dataset.

---

## 0. Results (held-out test split, best model)

The best model is the **tiled / per-key ROI** architecture at full key resolution
(`configs/tiled_best.yaml`, run `tiled_best_v2`), evaluated on the 9 held-out
**test** recordings with calibrated decode thresholds (onset=0.60, frame=0.50):

| Metric | Score | Meaning |
|---|---|---|
| **Note F1 (onset + pitch)** | **0.84** | correct pitch, onset within 50 ms |
| **Note F1 (onset + offset + pitch)** | **0.69** | also correct key release (strict overall) |
| **Frame F1 (pitch-time grid)** | **0.84** | per-frame, per-key accuracy |
| **MIDI note agreement** | **86 %** | notes shared with the reference MIDI (pitch+onset, 50 ms) |

Validation (used for model selection + threshold calibration): onset F1 0.77,
onset+offset F1 0.68, frame F1 0.79. Three independent measures (mir_eval note
scores, frame scores, and raw MIDI overlap) all agree — the numbers are real.

> All scores are **video-only** (no audio) on recordings the model never trained on.

---

## 1. Pipeline overview

```
video frame (1920x1080)
      │  perspective warp using the 4 keyboard corners in metadata_v2.json
      ▼
rectified keyboard strip (W x H, e.g. 1408 x 112)
      │  per-frame 2D CNN encoder
      ▼
   ┌── "strip" arch: one global feature per frame ──┐
   └── "tiled" arch: one ROI feature PER KEY (88) ──┘   <-- best
      │  BiGRU (temporal)
      ▼
      ├──► onset head (88)  ──► note starts
      ├──► frame head (88)  ──► key-held roll
      └──► velocity head (88, optional/off)
      │  Onsets-and-Frames decoding (onset rising-edge + frame sustain)
      ▼
note events (pitch, onset, offset, velocity)  ──►  .mid
```

### Key design decisions
- **Visual only** — the audio modality is ignored entirely.
- **Tiled / per-key ROI model** (`model.arch=tiled`). The keyboard strip is split
  width-wise into one region per key, so every key gets its own feature and
  temporal stream. At full key resolution (`keyboard.warp_width=1408` → 88 feature
  columns, one per key) this **roughly doubled F1** vs. the original global "strip"
  model. This is the winning configuration.
- **Offset = `key_offset`** (finger physically leaving the key). The TSV's
  `frame_offset` includes sustain-pedal tails a camera cannot see, so it is the
  wrong target for a vision model. Configurable via `labels.offset_field`.
- **Velocity = constant** (`decode.default_velocity`, default 80). A velocity head
  exists but is disabled (`model.use_velocity: false`).
- **Splits**: trains on `train` + `ext-train`, validates on `valid`, tests on
  `test`. `special(blurry)` and `special(4hands)` are excluded by default. One
  corrupt video (`2024-09-05_21-37-08`) is excluded via `data.exclude_records`.

---

## 2. Servers & install

The project runs on two servers; the **same code and configs** work on both,
because the active machine is auto-detected by hostname (see *Server profiles*).

| Profile | Server (hostname) | Device | Code path | Dataset path |
|---|---|---|---|---|
| **dib** | `gondor` | **GPU** (RTX 3090) | `/home/achatzigiannis/tivit_vam_v2_repo/vision_transcription` | `/raid_storage/data_achatzigiannis/PianoVAM_v1.0` |
| **dit** | `vdcloud` | **CPU only** | `/home/mkoziri/tasos/tivit_vam_v2_repo/vision_transcription` | `/home/mkoziri/datasets/PianoVAM_v1.0` |

Training the full model is done on the **GPU server (dib)**; the CPU server (dit)
is fine for evaluation, visualisation and small tests.

### Install (run once per server)
```bash
cd <code path for this server>/vision_transcription
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# PyTorch: pick the build that matches the server
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # GPU server (dib)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU server (dit)

pip install -r requirements.txt
pip install -e .
```
> `decord` needs a Linux wheel (not available on Windows). A local Windows machine
> is only for editing code; all data/training runs on the servers.

### Server profiles (how the machine is chosen)
`configs/*.yaml` contain a `profiles:` block with `dit` and `dib` entries (paths +
device). The active one is picked, highest priority first:
1. CLI override: `profile=dib`
2. Environment variable: `export PIANOVAM_SERVER=dib`
3. `profile: auto` (default) → matched against the hostname (`gondor`→dib, `vdcloud`→dit)

So on either server you usually type nothing extra — it configures itself. Confirm with:
```bash
python -c "from pianovam_vision.config import load_config as L; c=L('configs/tiled_best.yaml',[]); print('profile:',c.get('_active_profile'),'device:',c['train']['device'],'root:',c['data']['root'])"
```

---

## 3. The full experiment — exact commands

All commands are run from `<code path>/vision_transcription` with `.venv` active.
The best experiment is defined once in **`configs/tiled_best.yaml`** (tiled model,
`warp_width=1408`, 20 epochs, the corrupt video excluded).

### 3a. Verify the data
```bash
python -m pianovam_vision.check_data --config configs/tiled_best.yaml
```

### 3b. Train (GPU server) — 20 epochs, crash-proof
```bash
CONFIG=configs/tiled_best.yaml bash scripts/run_experiment.sh tiled_best_v2 tiled train.num_workers=0
```
- Results are written to `runs/tiled_best_v2/` (`best.pt`, `last.pt`, `train.log`, `config.json`).
- **Resumable:** if it ever stops, run the **exact same command** again — it
  continues from `runs/tiled_best_v2/last.pt` (nothing is lost). The launcher also
  refuses to start a duplicate if one is already running.
- `train.num_workers=0` keeps it stable/unattended (no data-loader crashes).

**Monitor:**
```bash
bash scripts/status.sh                 # one-line status of every run
tail -f runs/tiled_best_v2/train.log   # live progress
```
**Stop:** `kill $(cat runs/tiled_best_v2/run.pid)`

### 3c. Evaluate on validation (note-level, mir_eval)
```bash
python -m pianovam_vision.evaluate --config configs/tiled_best.yaml \
    --checkpoint runs/tiled_best_v2/best.pt --split valid \
    --save_midi out/tiled_best_v2 train.max_frames_per_record=0
```

### 3d. Calibrate the decode thresholds on validation
```bash
python -m pianovam_vision.calibrate --config configs/tiled_best.yaml \
    --checkpoint runs/tiled_best_v2/best.pt --split valid --target full_f1
```
Prints the best `onset_threshold` / `frame_threshold` (for our best model: **0.60 / 0.50**).

### 3e. FINAL result — test split with the calibrated thresholds
```bash
python -m pianovam_vision.evaluate --config configs/tiled_best.yaml \
    --checkpoint runs/tiled_best_v2/best.pt --split test \
    --save_midi out/tiled_best_v2_test \
    train.max_frames_per_record=0 decode.onset_threshold=0.60 decode.frame_threshold=0.50
```
> **Command tip:** put all `--flags` first, then all bare `key=value` overrides
> **together at the end** (argparse rejects overrides sandwiched between flags).

### 3f. Compare predicted MIDI vs. the dataset MIDI
```bash
python -m pianovam_vision.compare_midi --config configs/tiled_best.yaml \
    --pred_dir out/tiled_best_v2_test
```
Reports the % of notes in common (same pitch, onset within 50 ms).

> **Methodology:** thresholds are calibrated on **valid** and reported on **test**;
> the test split is used only once, at the very end — so 3e is the honest result.

---

## 4. Where to find the results (paths on the GPU server)

Base: `/home/achatzigiannis/tivit_vam_v2_repo/vision_transcription/`

| What | Path |
|---|---|
| **Trained model + logs** | `runs/tiled_best_v2/` → `best.pt`, `last.pt`, `train.log`, `config.json` |
| **Predicted MIDI (test)** | `out/tiled_best_v2_test/*.mid` |
| **Predicted MIDI (valid)** | `out/tiled_best_v2/*.mid` |
| **Corner/sync preview images** | `preview/` (`*_raw.png`, `*_warp.png`) |
| **Keyboard + note-label images** | `preview_keys/*_keys.png` |
| **Left/Right-hand overlay images** | `preview_hands/*_hands.png` |

`config.json` inside each `runs/<name>/` folder is the **exact, reproducible
configuration** that produced that model.

**Download images/MIDI to a laptop** (run locally):
```bash
scp -r achatzigiannis@gondor:~/tivit_vam_v2_repo/vision_transcription/preview_keys ./
scp -r achatzigiannis@gondor:~/tivit_vam_v2_repo/vision_transcription/out/tiled_best_v2_test ./
```

---

## 5. Visualisation & sanity tools

### 5a. Corner + sync check (do this before trusting any run)
```bash
python -m pianovam_vision.preview --config configs/tiled_best.yaml \
    --record_time 2024-02-14_19-10-09 --time 30.0 --out_dir preview/
```
`*_warp.png` draws green lines on the pitches the labels say are held — they should
land on visibly pressed keys.

### 5b. Keyboard with per-key borders + note names
```bash
python -m pianovam_vision.draw_keyboard --config configs/tiled_best.yaml \
    --record_time 2024-02-14_19-10-09 --busiest --out_dir preview_keys/   # one recording
python -m pianovam_vision.draw_keyboard --config configs/tiled_best.yaml \
    --time 20 --out_dir preview_keys/                                     # all recordings
```
Draws white-key borders (red), black keys (cyan) and note labels (`C4`, `C#`, …).
`--busiest` jumps to the moment with the most simultaneous notes.

### 5c. Left / Right hand assignment (uses the `Handskeleton/` data)
```bash
python -m pianovam_vision.draw_hands --config configs/tiled_best.yaml \
    --record_time 2024-02-14_19-10-09 --busiest --out_dir preview_hands/
```
For each pressed key it colours the hand that played it (🔵 Left, 🔴 Right) using the
MediaPipe hand landmarks, and overlays the fingertips — a visual check of which hand
hit which pitch. *(This is analysis/verification; the transcription model itself does
not yet predict hands — see Roadmap.)*

### 5d. Transcribe a single video → MIDI
```bash
python -m pianovam_vision.infer --config configs/tiled_best.yaml \
    --checkpoint runs/tiled_best_v2/best.pt \
    --record_time 2024-02-14_19-10-09 --output out/2024-02-14_19-10-09.mid
```

### 5e. Logic tests (no GPU/video needed)
```bash
python -m pytest tests/ -q
```

---

## 6. Repository layout

```
vision_transcription/
├── configs/
│   ├── default.yaml            # base config (strip model), server profiles
│   └── tiled_best.yaml         # pinned best experiment (tiled, warp_width=1408)
├── scripts/
│   ├── run_experiment.sh       # launch/resume a training run into runs/<name>/
│   └── status.sh               # one-line status of every run
├── pianovam_vision/
│   ├── config.py               # YAML + dotted CLI overrides + server profiles
│   ├── metadata.py             # parse metadata_v2.json (splits + corner points)
│   ├── keyboard.py             # perspective warp + real piano key geometry
│   ├── labels.py               # TSV parsing + onset/frame/velocity target rolls
│   ├── video.py                # decord reader + on-the-fly warping
│   ├── dataset.py              # clip Dataset (skips corrupt clips)
│   ├── model.py                # strip + tiled (per-key ROI) architectures
│   ├── decode.py               # roll → note events
│   ├── midi_io.py              # write .mid
│   ├── metrics.py              # frame PRF + mir_eval note scores
│   ├── train.py                # training loop (+ --resume)
│   ├── evaluate.py             # note-level evaluation on a split
│   ├── calibrate.py            # tune decode thresholds on valid
│   ├── infer.py                # single-video → MIDI
│   ├── compare_midi.py         # predicted vs. dataset MIDI overlap
│   ├── preview.py              # corner/sync sanity tool
│   ├── draw_keyboard.py        # per-key borders + note labels
│   ├── draw_hands.py           # left/right-hand overlay (Handskeleton)
│   ├── hands.py                # derive Left/Right per note from Handskeleton
│   └── check_data.py           # dataset layout check
└── tests/test_pipeline.py
```

---

## 7. Assumptions & things to confirm
1. **Time alignment.** Video time `t=0` = TSV/MIDI time `t=0`. Run `preview` on a
   few recordings to confirm sync before a long run.
2. **Corner points bound the 88 keys** edge-to-edge (A0…C8). The `draw_keyboard`
   overlay is the visual check per recording.
3. **Frame rate.** Video is 60 fps; labels/inference run at `labels.fps` (30). Onset
   tolerance in evaluation is 50 ms.
4. **Excluded data.** `special(blurry)`, `special(4hands)`, and the corrupt video
   `2024-09-05_21-37-08` are excluded (`data.exclude_records`).

## 8. Tuning knobs that matter most
| Goal | Knob |
|---|---|
| Sharper per-key features | `keyboard.warp_width=1408` (one column per key) |
| More training data | raise `train.max_frames_per_record` (600 → 1800 → 0 = full) |
| Stability (no data-loader crash) | `train.num_workers=0` |
| Too many spurious notes | raise `decode.onset_threshold` (calibrate) |
| Missing/short notes | lower `decode.frame_threshold` / `decode.min_duration_s` |
| GPU/CPU device | set by server profile (`train.device`) |

## 9. Roadmap / next steps
- **More epochs + more data** (`max_frames_per_record` toward full videos) to push
  F1 further — the biggest remaining lever.
- **Left/right-hand head**: predict the playing hand per key (ground truth derived
  from `Handskeleton/` via `hands.py`), reported as a separate hand-accuracy metric.
- **Skeleton as an input** for occluded-key reasoning (may modestly help recall).
- **Improve offsets** (the weakest metric): richer temporal/frame modelling.
- **Velocity head** experiments (`model.use_velocity=true`).
