# PianoVAM — Video-only Piano Transcription

Recognise pressed piano keys (**onset**, **key-release offset**, **pitch**) directly
from the keyboard **video** and export a **MIDI** file. **Audio is never used** —
the model only looks at the keys.

This implements a visual *Onsets-and-Frames* network (the acoustic spectrogram is
replaced by a perspective-rectified image of the keyboard), trained on the
PianoVAM v1.0 labels.

---

## 1. Pipeline overview

```
video frame (1920x1080)
      │  perspective warp using the 4 keyboard corners in metadata_v2.json
      ▼
rectified keyboard strip (W x H, e.g. 896 x 112)
      │  per-frame 2D CNN encoder
      ▼
feature sequence  ──► BiGRU (temporal) ──► onset head (88)
                                       └──► frame head (88)   ──► press roll
                                       └──► velocity head (88, optional/off)
      │  Onsets-and-Frames decoding (onset rising-edge + frame sustain)
      ▼
note events (pitch, onset, offset, velocity)  ──►  .mid
```

### Key design decisions (agreed up front)
- **Visual only** — audio modality is ignored entirely.
- **Offset = `key_offset`** (the finger physically leaving the key). The TSV's
  `frame_offset` includes sustain-pedal tails, which a camera *cannot* see, so it
  is the wrong target for a vision model. Configurable via `labels.offset_field`.
- **Velocity = constant** (`decode.default_velocity`, default 80). A velocity
  **head** exists in the model but is disabled (`model.use_velocity: false`);
  flip it on later to experiment with visual velocity regression.
- **Splits**: trains on `train` + `ext-train`, validates on `valid`, tests on
  `test`. `special(blurry)` and `special(4hands)` are excluded by default.

---

## 2. Install (on the GPU server)

```bash
cd /home/achatzigiannis/<project>/vision_transcription
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # install the CUDA torch build for your server
pip install -e .                         # optional, enables `import pianovam_vision` anywhere
```

> `decord` needs a Linux wheel (it is not available on Windows). The local Windows
> machine is only for editing code; all data/training runs on the server.

Set the dataset path once (edit `configs/default.yaml` → `data.root`) or override
on every command with `data.root=/raid_storage/data_achatzigiannis/PianoVAM_v1.0`.

---

## 3. Step-by-step usage

### 3a. Verify the data is wired up
```bash
python -m pianovam_vision.check_data --config configs/default.yaml
```
Lists split counts and flags any missing video/TSV files.

### 3b. **Sanity-check corners + sync (do this first!)**
```bash
python -m pianovam_vision.preview --config configs/default.yaml \
    --record_time 2024-02-14_19-10-09 --time 30.0 --out_dir preview/
```
Open the two PNGs. In `*_warp.png`, the green vertical lines mark the pitches the
TSV says are held at that instant — they should land on visibly pressed keys. If
they don't, either the corner points or the video↔label time alignment is off
(see *Assumptions* below) and training will not work until that's fixed.

### 3c. Train
```bash
python -m pianovam_vision.train --config configs/default.yaml \
    train.out_dir=runs/exp1
```
Checkpoints `best.pt` / `last.pt` land in `train.out_dir`. Validation prints
frame- and onset-level F1 each epoch.

Quick smoke test (tiny, runs in minutes) before a full run:
```bash
python -m pianovam_vision.train --config configs/default.yaml \
    train.max_frames_per_record=600 train.epochs=2 train.batch_size=2 \
    train.num_workers=4 train.out_dir=runs/smoke
```

### 3d. Evaluate (note-level F1, mir_eval)
```bash
python -m pianovam_vision.evaluate --config configs/default.yaml \
    --checkpoint runs/exp1/best.pt --split test --save_midi out/test
```
Reports onset-F1 and onset+offset-F1 vs the `key_offset` ground truth.

### 3e. Transcribe a single video → MIDI
```bash
python -m pianovam_vision.infer --config configs/default.yaml \
    --checkpoint runs/exp1/best.pt \
    --record_time 2024-02-14_19-10-09 \
    --output out/2024-02-14_19-10-09.mid
```
For a video outside the dataset, pass `--video path.mp4 --corners "x,y x,y x,y x,y"`
(corner order: **LT RT RB LB**).

### 3f. Run the logic tests (no GPU/video needed)
```bash
python -m pytest tests/ -q
```

---

## 4. Repository layout

```
vision_transcription/
├── configs/default.yaml          # all hyper-parameters & paths
├── pianovam_vision/
│   ├── config.py                 # YAML + dotted CLI overrides
│   ├── metadata.py               # parse metadata_v2.json (splits + corner points)
│   ├── keyboard.py               # perspective warp, pitch→column helpers
│   ├── labels.py                 # TSV parsing + onset/frame/velocity target rolls
│   ├── video.py                  # decord reader + on-the-fly warping
│   ├── dataset.py                # clip Dataset (frames + targets)
│   ├── model.py                  # CNN encoder + BiGRU + onset/frame/velocity heads
│   ├── decode.py                 # roll → note events
│   ├── midi_io.py                # write .mid
│   ├── metrics.py                # frame PRF + mir_eval note scores
│   ├── train.py / infer.py / evaluate.py
│   ├── preview.py                # corner/sync sanity tool
│   └── check_data.py             # dataset layout check
└── tests/test_pipeline.py
```

---

## 5. Assumptions & things to confirm

1. **Time alignment.** We assume video time `t=0` corresponds to TSV/MIDI time
   `t=0`. The dataset README notes the *Sep 04–05* videos had a sync issue that
   was **fixed**; this code assumes you use the corrected videos. Always run the
   `preview` step on a few recordings per performer to confirm sync before a long
   training run. If a constant offset exists, add a per-record `time_offset` (easy
   to thread through `WarpedVideo` / `build_target_rolls`).
2. **Corner points bound the 88 keys** edge-to-edge. The model learns the
   pixel→pitch mapping from data, so this need only be approximately true; the
   `pitch_to_column` overlay in `preview` is the visual check.
3. **`jiwoo` recordings** have the upper body blurred (keyboard/hands unaffected),
   and live in `special(blurry)` — excluded by default. Add them to
   `data.train_splits` if you want to use them.
4. **Frame rate.** Video is 60 fps; we label/infer at `labels.fps` (default 30,
   i.e. every other frame). Lower fps = faster + coarser onset timing. Onset
   tolerance in eval is 50 ms.

---

## 6. Tuning knobs that matter most

| Goal | Knob |
|---|---|
| GPU OOM | lower `train.clip_len`, `train.batch_size`, or `keyboard.warp_height` |
| Slow data loading | raise `train.num_workers`; consider pre-caching warped frames |
| Too many spurious notes | raise `decode.onset_threshold` |
| Missing/short notes | lower `decode.frame_threshold`, lower `decode.min_duration_s` |
| Onsets under-detected in training | raise `train.onset_pos_weight` |
| Try visual velocity | `model.use_velocity=true` |

---

## 7. Roadmap / known next steps
- **Frame caching**: pre-warp frames to disk (memmap/LMDB) to remove per-epoch
  decode+warp cost — biggest training-speed win.
- **Velocity head** experiments (cue: key-press speed across consecutive frames).
- **Hand-occlusion handling** using the provided `Handskeleton/` MediaPipe data
  to mask/inform occluded keys.
- **Two-stage Audeo-style refinement** (a Roll→MIDI GRU/GAN) if the single
  end-to-end network's offsets are noisy.
