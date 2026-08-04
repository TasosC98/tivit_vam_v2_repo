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

```bash
# 0. Sanity-check the layout (split counts + which files are present/missing)
python -m pianovam_vision.check_data --config configs/pianoyt.yaml

# 1. VERIFY CORNERS + SYNC FIRST (see the gotcha below) on a few recordings
python -m pianovam_vision.preview --config configs/pianoyt.yaml \
    --record_time 100 --time 30 --snap_to_onset --out_dir preview/

# 2. Train / resume (same script as PianoVAM)
CONFIG=configs/pianoyt.yaml bash scripts/run_experiment.sh pianoyt_tiled tiled train.num_workers=0

# 3. Calibrate decode thresholds on the held-out valid split
python -m pianovam_vision.calibrate --config configs/pianoyt.yaml \
    --checkpoint runs/pianoyt_tiled/best.pt --split valid --target onset_f1

# 4. Evaluate on test with the calibrated thresholds
python -m pianovam_vision.evaluate --config configs/pianoyt.yaml \
    --checkpoint runs/pianoyt_tiled/best.pt --split test --save_midi out/pianoyt_test \
    decode.onset_threshold=<ot> decode.frame_threshold=<ft>
```

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
   exactly as PianoVAM does with `compare_midi`.
3. **No native validation split** — a deterministic 10% of train is relabelled
   `valid` (`data.valid_frac` / `data.valid_seed`). Set `valid_frac: 0.0` to disable.
