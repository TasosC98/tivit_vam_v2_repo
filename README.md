# KeyWise: video-only piano transcription with a per-key architecture

Code of the paper *KeyWise: Efficient Video-Only Piano Transcription with a Lightweight
Per-Key Architecture* (Anastasios Chatzigiannis, Natalia Panagou, Maria Koziri, University of
Thessaly).

KeyWise recovers the notes of a piano performance from a top-view video of the keyboard,
without the audio. Each frame is warped to a 1408 x 112 keyboard strip and cut into 88 equal
tiles, one region of interest (ROI) per key. A CNN encoder gives exactly one feature column per
key ROI, and one small BiGRU, shared by all 88 keys, follows each key over time and predicts,
for every frame, whether a note starts (onset) and whether the key is held (frame), and
optionally the note velocity. The model has 1.3 M parameters.

## Installation

Python 3.9 or newer. Install the PyTorch build that matches your GPU first
(https://pytorch.org), then:

```bash
pip install -r requirements.txt
python -m pytest tests -q          # about 2 minutes on a CPU, no data needed
```

`decord` is only needed to read the videos (building the strip cache, transcribing a new
video); training and evaluation read the strip cache. All commands below are run from the
repository root.

## Data

| Dataset | Put it at (or edit `data.root` in the config) | Contents used |
|---|---|---|
| PianoVAM | `data/PianoVAM` | `metadata_v2.json` (official v1.1 split + keyboard corners), `Video/*.mp4`, `TSV/*.tsv` |
| PianoYT | `data/PianoYT` | `pianoyt.csv`, `raw_videos/video_<id>.0.mp4`, `pianoyt_MIDI/audio_<id>.0.midi` |

Use the current PianoVAM labels: the TSV files were corrected on Hugging Face on 2026-03-18,
and the offset (key release) results need the corrected files. Recording
`2024-09-05_21-37-08` does not decode and is excluded. For PianoYT, `configs/pianoyt.yaml`
lists the seven training videos that were removed and holds out a fixed 10% of the training
videos (21) for validation.

For the comparison with V2N, its released PianoVAM test predictions are
`predicted_midi/pianovam_test` in https://github.com/yonghyunk1m/V2N.

## Reproducing the paper

| Paper | Config | Evaluation target |
|---|---|---|
| KeyWise on PianoVAM | `configs/pianovam.yaml` | `full_f1`, with `BASELINE_MIDI` = V2N |
| KeyWise + velocity | `configs/pianovam_velocity.yaml` | `full_f1`, with `BASELINE_MIDI` = V2N |
| Ablation: global variant | `configs/pianovam_global.yaml` | `full_f1`, with `BASELINE_MIDI` = V2N |
| Ablation: Global-88 | `configs/pianovam_global88.yaml` | `full_f1`, with `BASELINE_MIDI` = V2N |
| PianoYT, trained on PianoYT only | `configs/pianoyt.yaml` | `onset_f1` |
| PianoYT, pre-trained on PianoVAM | `configs/pianoyt_finetune.yaml` | `onset_f1` |
| PianoYT, zero-shot | a PianoVAM model with `DATA_CONFIG=configs/pianoyt.yaml` | `onset_f1` |

Every model was trained three times, with `train.seed` 1234, 1235 and 1236 (the zero-shot
result is one run). The paper's models were trained on one 24 GB GPU (RTX 3090); a training
run takes about a day.

**1. Strip cache.** Decodes every video once (720p), warps the keyboard and stores the strips
as JPEG, so that training does not decode videos (roughly 15-40 KB per frame):

```bash
python -m pianovam_vision.build_cache --config configs/pianovam.yaml --workers 6
python -m pianovam_vision.build_cache --config configs/pianoyt.yaml --workers 6
```

**2. Training.** `best.pt` keeps the epoch with the best validation onset F1:

```bash
for s in 1 2 3; do
  python -m pianovam_vision.train --config configs/pianovam.yaml \
      train.seed=$((1233 + s)) train.out_dir=runs/pianovam_s$s
done
# the same with pianovam_velocity / pianovam_global / pianovam_global88 / pianoyt, and
# the fine-tuning, where run s starts from PianoVAM run s:
python -m pianovam_vision.train --config configs/pianoyt_finetune.yaml \
    train.seed=1234 train.init_from=runs/pianovam_s1/best.pt train.out_dir=runs/finetune_s1
```

**3. Evaluation.** `scripts/evaluate.sh` chooses the two decoding thresholds on the validation
split, scores the test split with exactly those thresholds, writes the predicted MIDI and,
with `BASELINE_MIDI`, also scores V2N's MIDI with the same code:

```bash
BASELINE_MIDI=/path/to/V2N/predicted_midi/pianovam_test \
    bash scripts/evaluate.sh pianovam_s1 configs/pianovam.yaml runs/pianovam_s1/best.pt full_f1
bash scripts/evaluate.sh pianoyt_s1 configs/pianoyt.yaml runs/pianoyt_s1/best.pt onset_f1
DATA_CONFIG=configs/pianoyt.yaml \
    bash scripts/evaluate.sh zeroshot_s1 configs/pianovam.yaml runs/pianovam_s1/best.pt onset_f1
```

Results go to `results/<name>_test.txt` (mean scores), `results/<name>_test.csv` (per recording)
and `results/<name>_vs_baseline.csv` (KeyWise and V2N per recording, scored with the same
code; the PianoVAM numbers of the paper). Onset F1 is reported at 50 and 100 ms; "++Off"
requires the offset within the same tolerance, "++Vel" a correct velocity under V2N's
convention (velocity / 127 in mir_eval, i.e. within 12.7 MIDI steps after mir_eval's linear fit
per recording). Frame F1 uses a 60 Hz grid in `_vs_baseline.csv`, as in V2N (the model's 30 Hz
grid in `_test.txt`).

**4. Statistics, analyses and figures** (no GPU; the names are the evaluation names above):

```bash
python paper/compare_runs.py --results results --a pianovam_s1,pianovam_s2,pianovam_s3 --b V2N
python paper/compare_runs.py --results results --a velocity_s1,velocity_s2,velocity_s3 --b V2N --metrics velocity
python paper/compare_runs.py --results results --a pianovam_s1,pianovam_s2,pianovam_s3 \
    --b global_s1,global_s2,global_s3
python paper/analyse_notes.py --config configs/pianovam.yaml \
    --system KeyWise=out/pianovam_s1_test,out/pianovam_s2_test,out/pianovam_s3_test \
    --system Global=out/global_s1_test,out/global_s2_test,out/global_s3_test \
    --system Global88=out/global88_s1_test,out/global88_s2_test,out/global88_s3_test \
    --system V2N=/path/to/V2N/predicted_midi/pianovam_test --out results/note_analysis.json
python paper/make_figures.py --results results --analysis results/note_analysis.json --out figures
```

`compare_runs.py` gives the mean and standard deviation over the runs, the exact two-sided
Wilcoxon test over the recordings (on the run means), the Holm-Bonferroni correction and 95%
bootstrap intervals. `analyse_notes.py` gives the error classes, the recall by register and key
colour, the timing and offset analyses and the per-key statistics.

**5. Size and speed.**

```bash
python -m pianovam_vision.speed --config configs/pianovam.yaml       # parameters, MACs, frames/s
python paper/speed_vs_v2n.py --v2n /path/to/V2N --json results/speed_vs_v2n.json
```

`speed_vs_v2n.py` builds V2N from its own code and times both models on the same GPU, each
with the overlap it uses at test time (KeyWise: 512-frame windows every 448 frames; V2N: 1 s
clips every 0.5 s) and without overlap.

**Other checks in the paper.** Fixed thresholds: add `decode.onset_threshold=0.5
decode.frame_threshold=0.5` (or 0.889 and 0.667, i.e. lambda / (1 + lambda) of the loss
weights 8 and 2) to `python -m pianovam_vision.evaluate`; last epoch: evaluate
`runs/<name>/last.pt` instead of `best.pt`.

## Transcribing a video

```bash
python -m pianovam_vision.infer --config configs/pianovam.yaml --checkpoint runs/pianovam_s1/best.pt \
    --video my_video.mp4 --corners "121,355 1839,345 1839,558 120,564" --output my_video.mid \
    decode.onset_threshold=0.90 decode.frame_threshold=0.50
```

`--corners` are the keyboard corners in the video frame (top-left, top-right, bottom-right,
bottom-left; left = A0); use the thresholds that calibration chose for that model.

## Repository

```
configs/            the experiments of the paper (see the table above)
pianovam_vision/    the package: data (metadata, labels, video, strip_cache, build_cache, dataset,
                    augment), model, train, inference and decoding (infer, decode, rolls),
                    calibrate, evaluate, metrics, score_midi, speed
scripts/evaluate.sh threshold calibration on valid + test evaluation (+ comparison with V2N)
paper/              statistics (compare_runs), note analyses, figures, speed comparison with V2N
tests/              unit tests and an end-to-end run on a tiny synthetic dataset
```
