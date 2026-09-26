# Running the experiment on PianoYT

This is the **same** visual (video-only) transcription pipeline as PianoVAM, run
on the [PianoYT](https://www.robots.ox.ac.uk/~vgg/research/sighttosound/) dataset
instead. The model / training / decoding / metrics are unchanged; only the
data-loading layer differs, selected by `data.format: pianoyt` in
[`configs/pianoyt.yaml`](configs/pianoyt.yaml). Results are directly comparable
to the PianoVAM `tiled_best` run.

## What differs from PianoVAM

| Aspect | PianoVAM | PianoYT |
|---|---|---|
| Metadata | `metadata_v2.json` | `pianoyt.csv` (`id, url, split, min_y, max_y, min_x, max_x`) |
| Recording id | datetime stem | integer id (`100`) |
| Splits | train/ext-train/valid/test | `1`=train, `3`=test; **valid is carved from train** (`data.valid_frac`, default 10%) |
| Keyboard region | 4 perspective corners | axis-aligned crop box → 4 corners (LT,RT,RB,LB) |
| Labels | TSV (`key_offset` = visual release) | MIDI `audio_<id>.0.midi` (note-off offsets **include pedal**) |
| Video | local 1920×1080 @60fps | YouTube download, varying resolution/fps |

The loaders live in [`metadata.py`](pianovam_vision/metadata.py)
(`load_recordings_pianoyt`, `recordings_from_cfg`) and
[`labels.py`](pianovam_vision/labels.py) (`read_midi`, `read_reference`).

## Expected dataset layout on the server

```
<data.root>/
├── pianoyt.csv
├── pianoyt_MIDI/         audio_<id>.0.midi        (ground-truth labels)
└── raw_videos/           video_<id>.0.mp4         (downloaded YouTube videos)
```
Filenames/dirs are configurable: `data.video_dir` / `data.video_pattern` /
`data.midi_dir` / `data.midi_pattern` (patterns are `str.format` templates taking
`{id}`). Videos or MIDIs that are missing (removed from YouTube, etc.)
are **skipped automatically** at train/eval time — no need to list them in
`exclude_records`. Locally there are 228 rows and 228 MIDI files; videos must be
downloaded separately.

## Run it

**The full paper workflow (both datasets, cross-dataset, sync correction) is in
[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md).** The PianoYT-only essentials:

```bash
# 0. Layout + per-video resolution/fps/duration/crop checks
python -m pianovam_vision.check_data --config configs/yt_full.yaml --probe

# 1. VERIFY CROPS + SYNC FIRST: every video, keys struck in the last 100 ms in green
python -m pianovam_vision.draw_keyboard --config configs/yt_full.yaml \
    --busiest --onsets_only --sheet 10 --out_dir preview_keys_yt/

# 2. Build the strip cache once, then train / resume (full videos, augmentation)
python -m pianovam_vision.build_cache --config configs/yt_full.yaml --workers 6
CONFIG=configs/yt_full.yaml bash scripts/run_experiment.sh yt_full tiled

# 3. Calibrate decode thresholds on the held-out valid split
python -m pianovam_vision.calibrate --config configs/yt_full.yaml \
    --checkpoint runs/yt_full/best.pt --split valid --target onset_f1

# 4. Evaluate on test (whole videos; prints onset F1 at 50 AND 100 ms)
python -m pianovam_vision.evaluate --config configs/yt_full.yaml \
    --checkpoint runs/yt_full/best.pt --split test --csv results/yt_full_test.csv \
    decode.onset_threshold=<ot> decode.frame_threshold=<ft>
```

`configs/pianoyt.yaml` is the original 20-s-per-video recipe (run
`pianoyt_tiled`); `yt_full.yaml` inherits from it.

## Gotchas specific to PianoYT

1. **Crop box is in native-resolution pixels.** The `min_y…max_x` box assumes the
   frame resolution the PianoYT authors extracted (some boxes exceed 1080 px tall,
   so the source was larger than 1080p). `WarpedVideo` probes each video's native
   size and scales the corners, so what matters is that the **downloaded video's
   native resolution matches the box's coordinate system**. If you downloaded at a
   different resolution, the warp will be off. **Always run `preview` first** — the
   green pitch lines must land on pressed keys.
2. **Offsets include sustain pedal.** PianoYT labels are MIDI extracted from audio
   (Onsets-and-Frames), so note-off times carry pedal tails a camera can't see.
   Report/optimise **onset+pitch (`onset_f1`)**, not `full_f1` (which will read low),
   exactly as PianoVAM does with `compare_midi`. The published PianoYT numbers
   ("Pay Attention to the Keys": S2S 0.64, V2R 0.64, ViT 0.68) use a **100 ms**
   onset tolerance — compare against our `@100` column, not `@50`.
3. **No native validation split** — a deterministic 10% of train is relabelled
   `valid` (`data.valid_frac` / `data.valid_seed`). Set `valid_frac: 0.0` to disable.
4. **Audio/video sync.** Labels follow each video's audio track; a video whose
   sound and picture drift apart has shifted labels. `sync_check` estimates the
   per-video shift with a trained model and writes corrections for
   `data.label_offsets` (see `EXPERIMENT_PLAN.md`, Phase 4).
