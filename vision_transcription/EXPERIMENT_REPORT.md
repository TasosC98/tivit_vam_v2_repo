# PianoVAM Experiment Report

*A self-contained refresher: what we built, the result, how to reproduce it from
scratch, and how to make it better. Read top-to-bottom.*

---

## 1. What this project is (in one paragraph)

We transcribe piano music **from video only** — no audio. A camera looks down at
the keyboard; the model watches the keys and outputs which notes were played
(**pitch**), **when they started** (onset), and **when the finger left the key**
(offset). The final output is a **MIDI file**. Technically it's a visual version
of the classic *Onsets-and-Frames* transcription network: instead of an audio
spectrogram, the input is a perspective-corrected image of the keyboard, and the
model predicts, for each of the 88 keys and each video frame, "is this key being
pressed?" and "did a note just start here?".

Dataset: **PianoVAM v1.0** (107 recordings with video, MIDI, per-note TSV labels,
keyboard corner coordinates, and MediaPipe hand skeletons).

---

## 2. The result (our best experiment — the good outcome)

Best model = **tiled / per-key ROI architecture at full key resolution**
(config `configs/tiled_best.yaml`, run name `tiled_best_v2`). Evaluated on the
**9 held-out test recordings** (never seen in training) with calibrated decode
thresholds (onset=0.60, frame=0.50):

| Metric | Score | Plain-English meaning |
|---|---|---|
| **Note F1 (onset + pitch)** | **0.84** | correctly finds a note (right key, onset within 50 ms) 84% of the time |
| **Note F1 (onset + offset + pitch)** | **0.69** | also gets the release right — the strict "overall" score |
| **Frame F1 (pitch-time grid)** | **0.84** | per-frame, per-key "is this key down" accuracy |
| **MIDI note agreement** | **86 %** | our predicted MIDI shares 86% of its notes with the dataset's MIDI |

Validation numbers (used to pick the model + calibrate): onset F1 0.77, overall
0.68, frame F1 0.79.

**Why we trust these:** three *independent* measurements agree —
mir_eval note scoring (0.84), frame scoring (0.84), and raw MIDI overlap (0.86).
For **video-only** transcription, 0.84 note F1 is a strong result.

---

## 3. What every number means (so you can explain it)

- **Onset** = the moment a note *starts*. Sparse and timing-critical → the hard part.
- **Offset** = the moment the key is *released*. Hardest to see from video.
- **Frame** = "is key K held down in this frame?" (sustain) → the easier part.
- **Precision** = of the notes the model predicted, what fraction are correct
  (low precision = too many false notes).
- **Recall** = of the real notes, what fraction the model found
  (low recall = misses).
- **F1** = the single balanced summary = harmonic mean of precision and recall.
- **50 ms tolerance** = a predicted onset counts as correct if it's within 50 ms of
  the true onset (standard in music transcription; our video is 30 fps = 33 ms/frame).

So "Note F1 (onset+pitch) = 0.84" means: *matching the right key at the right time
(±50 ms), the model and the ground truth agree 84% of the time.*

---

## 4. How the system works (plain words)

```
video frame (1920x1080)
  → warp the keyboard flat using its 4 corner points (from metadata_v2.json)
  → rectified keyboard strip (1408 x 112 pixels)
  → CNN splits the strip into 88 vertical ROIs, one per key   ("tiled" model)
  → a BiGRU looks across time
  → two outputs per key: "note started?" (onset) and "key held?" (frame)
  → decode into note events (pitch, onset, offset)
  → write MIDI
```

**Key idea that won:** giving the model **one feature column per key** (by making
the warped strip 1408 px wide → exactly 88 columns) roughly **doubled** the F1
versus a coarse global model. That's the `tiled` architecture at `warp_width=1408`.

---

## 5. The two servers

| Profile | Hostname | Device | Code path | Dataset path |
|---|---|---|---|---|
| **dib** | `gondor` | **GPU** (RTX 3090) | `/home/achatzigiannis/tivit_vam_v2_repo/vision_transcription` | `/raid_storage/data_achatzigiannis/PianoVAM_v1.0` |
| **dit** | `vdcloud` | CPU only | `/home/mkoziri/tasos/tivit_vam_v2_repo/vision_transcription` | `/home/mkoziri/datasets/PianoVAM_v1.0` |

The code **auto-detects** which server it's on (by hostname) and uses the right
paths + device. You normally don't specify anything. **Train on the GPU server (dib).**

---

## 6. How to run the whole experiment FROM SCRATCH

All commands run from the code path, with the virtualenv active. Do these on the
**GPU server (gondor / dib)**.

### Step 0 — set up (only if the venv is missing)
```bash
cd /home/achatzigiannis/tivit_vam_v2_repo/vision_transcription
git pull                                   # get the latest code
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```
If the venv already exists, just:
```bash
cd /home/achatzigiannis/tivit_vam_v2_repo/vision_transcription
source .venv/bin/activate
git pull
```

### Step 1 — check the data is found
```bash
python -m pianovam_vision.check_data --config configs/tiled_best.yaml
```
Expect `... (107 recordings)` and `OK`.

### Step 2 — TRAIN (fresh, 20 epochs)
```bash
CONFIG=configs/tiled_best.yaml bash scripts/run_experiment.sh tiled_best_v2 tiled train.num_workers=0
```
- Writes everything to `runs/tiled_best_v2/`.
- **It resumes automatically:** if it stops for any reason, run the **exact same
  command again** and it continues from where it left off (`last.pt`).
- To start **completely fresh** instead of resuming, use a new name (e.g.
  `tiled_best_v3`) or delete the folder first: `rm -rf runs/tiled_best_v2`.

**Watch it:**
```bash
bash scripts/status.sh                 # summary of every run
tail -f runs/tiled_best_v2/train.log   # live; Ctrl+C stops watching, NOT training
```
**Stop it:** `kill $(cat runs/tiled_best_v2/run.pid)`

### Step 3 — EVALUATE on validation (note-level)
```bash
python -m pianovam_vision.evaluate --config configs/tiled_best.yaml \
    --checkpoint runs/tiled_best_v2/best.pt --split valid \
    --save_midi out/tiled_best_v2 train.max_frames_per_record=0
```

### Step 4 — CALIBRATE the thresholds on validation
```bash
python -m pianovam_vision.calibrate --config configs/tiled_best.yaml \
    --checkpoint runs/tiled_best_v2/best.pt --split valid --target full_f1
```
It prints the best `onset_threshold` / `frame_threshold` (last time: **0.60 / 0.50**).
Use those values in Step 5.

### Step 5 — FINAL RESULT on the test split (with calibrated thresholds)
```bash
python -m pianovam_vision.evaluate --config configs/tiled_best.yaml \
    --checkpoint runs/tiled_best_v2/best.pt --split test \
    --save_midi out/tiled_best_v2_test \
    train.max_frames_per_record=0 decode.onset_threshold=0.60 decode.frame_threshold=0.50
```
This is the number you report.

### Step 6 — MIDI comparison (extra confirmation)
```bash
python -m pianovam_vision.compare_midi --config configs/tiled_best.yaml \
    --pred_dir out/tiled_best_v2_test
```

### (Optional) pictures for the report
```bash
python -m pianovam_vision.draw_keyboard --config configs/tiled_best.yaml \
    --record_time 2024-02-14_19-10-09 --busiest --out_dir preview_keys/
python -m pianovam_vision.draw_hands --config configs/tiled_best.yaml \
    --record_time 2024-02-14_19-10-09 --busiest --out_dir preview_hands/
```

---

## 7. Where the outputs live (on the GPU server)

Base: `/home/achatzigiannis/tivit_vam_v2_repo/vision_transcription/`

| What | Path |
|---|---|
| Trained model + logs | `runs/tiled_best_v2/` (`best.pt`, `last.pt`, `train.log`, `config.json`) |
| Predicted MIDI (test) | `out/tiled_best_v2_test/*.mid` |
| Predicted MIDI (valid) | `out/tiled_best_v2/*.mid` |
| Keyboard label images | `preview_keys/*_keys.png` |
| Left/right-hand images | `preview_hands/*_hands.png` |
| Corner/sync check images | `preview/*_warp.png` |

`best.pt` = best model (highest validation frame-F1). `config.json` in each run
folder = the exact settings that produced it (fully reproducible).

To copy pictures/MIDI to your laptop:
```bash
scp -r achatzigiannis@gondor:~/tivit_vam_v2_repo/vision_transcription/out/tiled_best_v2_test ./
```

---

## 8. IMPORTANT gotchas (things that bit us before)

1. **Use `train.num_workers=0`.** With parallel data workers the run crashes/hangs
   on this dataset (bad video frames + shared RAM). `num_workers=0` is stable and
   unattended. It's already in every command above.
2. **If training stops, just rerun the same command** — it resumes from `last.pt`.
   Don't start over.
3. **Never launch the same run twice** (it corrupts the folder). The launcher now
   blocks duplicates, but if you see two, `pkill -9 -f pianovam_vision.train` and
   relaunch once.
4. **Config file is `configs/tiled_best.yaml`** (a file). `tiled_best_v2` is the
   **run name** (a folder under `runs/`). Don't mix them up — there is no
   `tiled_best_v2.yaml`.
5. **Command arguments:** put all `--flags` first, then the bare `key=value`
   overrides **together at the end**, or argparse errors.
6. **One corrupt video** (`2024-09-05_21-37-08`) is excluded automatically.
7. **Evaluation runs on 9 videos, not 107** — that's correct: `valid` and `test`
   are the held-out splits; the other ~80 are training data.

---

## 9. Are we at the peak? How to get BETTER results (you have a month)

**We are NOT at the ceiling.** The 0.84 result came from a model trained on a
**20-second slice of each video** (`max_frames_per_record=600`) and possibly not all
20 epochs. With a month of GPU time, here is the priority list to push higher:

1. **Train on MORE data — the biggest lever.** Raise the amount of each video used:
   ```bash
   # more data per recording (1 minute instead of 20 s):
   CONFIG=configs/tiled_best.yaml bash scripts/run_experiment.sh tiled_more tiled \
       train.num_workers=0 train.max_frames_per_record=1800

   # or the whole videos (best data; slow — but you have a month):
   CONFIG=configs/tiled_best.yaml bash scripts/run_experiment.sh tiled_full tiled \
       train.num_workers=0 train.max_frames_per_record=0
   ```
   More material almost always improves recall and generalisation. Then re-run
   Steps 3–6 on the new run name.

2. **Make sure it actually completes 20 epochs.** Check the current model's epoch:
   ```bash
   python -c "import torch;c=torch.load('runs/tiled_best_v2/best.pt',map_location='cpu',weights_only=False);print('best epoch:',c['epoch'])"
   ```
   If it's small, resume until it finishes 20 (or raise `train.epochs=40`).

3. **The offset score (0.69) is the weak point.** Detecting exactly when a finger
   leaves a key is genuinely hard from video. This is the main thing holding the
   strict overall score down — a good research direction.

4. **New capability (not more F1): left/right-hand prediction.** The dataset has
   MediaPipe hand skeletons; we built the tools to derive which hand played each
   note (`draw_hands`, `hands.py`). Adding a hand-prediction head would give a
   **new result** (hand accuracy) for the thesis — separate from transcription F1.

### Suggested plan for the month
1. Verify current model finished 20 epochs (Step in #2 above); if not, resume it.
2. Launch **`tiled_full`** (full videos) — the big-data run — and let it train for
   several days; monitor with `bash scripts/status.sh`.
3. When done, run Steps 3–6 on `tiled_full` and compare against `tiled_best_v2`.
4. If time remains, add the hand-prediction head for a bonus result.

---

## 10. One-line summary for your supervisor

> *A video-only, per-key Onsets-and-Frames model reaches **0.84 note F1
> (onset+pitch)**, **0.69 (onset+offset+pitch)**, and **0.84 frame F1** on the
> held-out PianoVAM test set — with **86%** note agreement to the reference MIDI —
> using no audio. Next: train on full-length videos and add left/right-hand
> attribution.*
