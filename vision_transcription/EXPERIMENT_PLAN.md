# Experiment plan — PianoVAM vs PianoYT (journal paper)

*Step-by-step. Every command runs on **gondor** from the PianoYT-branch clone,
which supports BOTH datasets:*

```bash
cd /home/achatzigiannis/tivit_pianoyt/vision_transcription
source /home/achatzigiannis/tivit_vam_v2_repo/vision_transcription/.venv/bin/activate
```

---

## 0. Where we stand

| | PianoVAM test, onset F1 @50 ms | @100 ms | PianoYT test, onset F1 @100 ms |
|---|---|---|---|
| **V2N** (arXiv 2608.03419, Aug 2026) | **94.7** | 97.0 | — |
| Li et al. 2024 (audio+video) | 94.2 | 97.4 | — |
| V2R / Audeo (re-trained by V2N) | 86.5 | 89.8 | 64 (R3-pretrained, by PPAN) |
| PPAN "Pay Attention to the Keys" (IJCAI'25) | 84.9 | 94.1 | **68** (R3-pretrained + fine-tuned) |
| S2S / Sight-to-Sound | 60.1 | 94.5 | 64 (R3-pretrained, by PPAN) |
| **ours, tiled_best_v2 / pianoyt_tiled** | 84.1 | 89.0 | 33 **@50 ms** — *@100 ms in Phase 0* |

**Verified head-to-head (2026-09-26).** Our `tiled_best_v2` test MIDI and V2N's
released test MIDI, scored by the same code (`score_midi`) on the same 9 test
videos: V2N's published numbers are reproduced exactly, our split is identical
to V2N's (`metadata_v2.json` = v1.1 `metadata.json`), and our reported scores
match only the corrected March-2026 labels — so the comparison is fair. V2N
wins on 8/9 videos (paired Wilcoxon p = 0.008):

| PianoVAM test | ours | V2N |
|---|---|---|
| Onset F1 @50 / @100 ms | 84.1 / 89.0 | 94.7 / 97.0 |
| Onset+offset F1 @50 ms, offset within 50 ms (V2N rule) | 63.8 | 89.6 |
| Frame F1 (60 Hz) | 83.2 | 90.5 |

Why ours is behind, in order of impact — **none of it is the architecture**:

1. **Data starvation.** `max_frames_per_record=600` = the first 20 s of every
   video, ~3% of PianoVAM's frames. PianoYT overfit after epoch 2 for the same
   reason (train loss 0.07, valid flat).
2. **Throughput.** The loader re-opened a 1080p video for every clip: 12 s per
   iteration with the GPU idle most of the time, 75 min per (tiny) epoch. Full
   data was impossible.
   → **strip cache** (`build_cache`): decode each video once.
3. **No augmentation** (PPAN's ablation: grayscale + augmentation took onset F
   from 0.18 to 0.39).
4. **Blurry input.** Decoding at 360p builds the 1408-px strip from ~570 real
   keyboard pixels (11 px per white key). 720p doubles that at no extra cost.
5. **Label bug (fixed):** notes after the covered window were stacked as fake
   onsets on its last frame, polluting validation → noisy `best.pt` choice.
   Reported test numbers were NOT affected.
6. **Protocol.** PianoYT literature numbers use a **100 ms** onset tolerance;
   we reported 50 ms. `evaluate` now prints both.

**Keep the model.** The per-key, weight-shared ROI design is the paper's hook:
V2N shows global models *collapse* across datasets (PianoVAM↔R3 onset F1
0–6%) and names geometry invariance as the open problem. A model that looks at
each key with the same weights is geometry-normalised by construction — the
cross-dataset experiments (E3/E4) test exactly that. Change the model only when
an experiment says so.

## 0b. Positioning against V2N (read this first)

V2N (Kim, Sohn, Nam, Lerch — the PianoVAM authors; ISMIR 2026; code, weights
and test predictions public) already publishes video-only transcription on
PianoVAM, on our split, and beats our current model everywhere. A paper whose
claim is "a new transcription model evaluated on PianoVAM" will not get
through. What V2N does **not** do — and itself lists as open problems — is
where this paper lives:

| Our contribution | V2N's status |
|---|---|
| Cross-dataset transfer with a keyboard-aligned, per-key model | "collapses everywhere"; cause: "the same column indexes a different key across datasets"; geometry invariance = open problem |
| PianoYT: clean (Disklavier) vs noisy (audio-derived) labels | PianoYT never used |
| Sync correction of noisy training labels (`sync_check`) | "joint sync alignment for noisy training data" = future work (they cross-correlated audio energy for R3) |
| Left/right-hand attribution per note | not addressed |

Consequences for the experiments:
- **The deciding experiment is E3** (zero-shot PianoVAM → PianoYT). If the
  per-key model transfers clearly better than V2N, that is the paper. Run it
  in Phase 0 with the old model, then with `vam_full`.
- **Run V2N itself on PianoYT** (their code is MIT) so the transfer claim is
  shown against the state of the art, not only argued. Needs PianoYT converted
  to their input format (corners + TSV) — ask for the converter.
- **Architecture vs crop ablation:** our `strip` (global) model vs `tiled`
  (per-key) under the same tight crop, on E3. Reviewers will ask whether the
  transfer comes from the per-key design or just from cropping tightly.
- **Hands need baselines:** PianoVAM's own fingering labels come from hand
  landmarks + a heuristic, and SKY-Piano (ISMIR 2026, same group) does
  fingering from motion capture. A hand head must beat (a) MediaPipe + nearest
  fingertip at the onset (`hands.py` on `origin/main`) and (b) a pitch-only
  split, especially where the hands cross.
- **PianoVAM stays in the paper** as the in-domain reference, reported against
  V2N honestly with `score_midi` (same code, their released MIDI). Closing the
  offset gap (64 vs 90) is V2N's territory: dedicated offset head, multi-task
  training, offset-guided decoding.
- **Speed matters:** V2N's future-work list overlaps our remaining angles.
  Post an arXiv preprint as soon as E3/E4 are solid.

## 1. The experiment matrix (answers "which dataset first?")

Not either/or: one model + one pipeline, run on both datasets. Each cell
answers a different question of the paper.

| ID | Train on | Test on | Question | Compare with |
|---|---|---|---|---|
| E1 | PianoVAM | PianoVAM | in-domain, clean Disklavier labels | V2N Table 1 |
| E2 | PianoYT | PianoYT | in-domain, noisy audio-derived labels | S2S/V2R 64, PPAN 68 (@100) |
| E3 | PianoVAM | PianoYT, **zero-shot** | does the per-key model transfer? | V2N: global models collapse |
| E4 | PianoVAM → fine-tune PianoYT | PianoYT | does clean pre-training help noisy data? | PPAN protocol (R3 → PianoYT) |
| E5 | PianoYT | PianoVAM, zero-shot | reverse transfer | — |
| E6 | PianoYT, **sync-corrected** labels | PianoYT | how much does label sync cost? | E2 |
| E7 | + hand head (PianoVAM) | PianoVAM | left/right hand accuracy | *nobody reports it* |

Order: **E1 first** (it is also the source model for E3/E4) with **E2 in
parallel**; then E3–E5 (evaluation only, hours); then E6; then E7.

---

## How the scripts work

Every long job runs **in the background** (`nohup`), so you can close PuTTY.
Each command prints where its log is; `tail -f <log>` follows it and **Ctrl+C
only stops the watching**, never the job. Results land in `results/`, logs in
`logs/`, predicted MIDI in `out/`, models in `runs/`.

| Script | Does | Phase |
|---|---|---|
| `scripts/phase0.sh` | re-scores the old models (incl. zero-shot), dataset checks, contact sheets | 0 |
| `scripts/build_caches.sh` | builds both strip caches (checks free disk first) | 1 |
| `scripts/run_experiment.sh` | trains / resumes one model (unchanged) | 2 |
| `scripts/evaluate_run.sh` | calibrates on valid → scores test with those thresholds | 3 |
| `scripts/status.sh` | one line per training run (unchanged) | any |

## Phase 0 — update the code, checks, free numbers (no training, a few hours)

```bash
git status                  # only __pycache__ files listed? -> git checkout -- .    (anything else: tell me first)
git pull                    # the new code (branch PianoYT)
python -m pytest tests/ -q  # expect "13 passed"  (no pytest? pip install pytest)
bash scripts/status.sh      # still running from before? stop it: kill $(cat runs/<name>/run.pid)
nvidia-smi; nproc; df -h /raid_storage
bash scripts/phase0.sh      # background job
tail -f logs/phase0.log
```
`phase0.sh` runs, in order: (1) the old PianoVAM model on the PianoYT test set
— **zero-shot, the key number**; (2) the old PianoVAM model on its own test set
at 50 and 100 ms; (3) the old PianoYT model at 50 and 100 ms; (4–5)
`check_data --probe` on both datasets (resolution, fps, hours, crop and
label-length problems); (6) the PianoYT contact sheets. If your old checkpoints
are elsewhere: `OLD_VAM=/path/best.pt OLD_YT=runs/<run>/best.pt bash scripts/phase0.sh`.

**Do not resume the old `pianoyt_tiled` run.** More epochs of 20-s clips only
overfit further; it stays in the paper as the "20 s / no augmentation" row.

**Look at the contact sheets** (`preview_keys_yt/sheet_*.png`, ≈20 images for
228 videos). Copy them to the laptop, e.g. in PowerShell with PuTTY's `pscp`:
`pscp achatzigiannis@gondor:/home/achatzigiannis/tivit_pianoyt/vision_transcription/preview_keys_yt/sheet_*.png .`
In each strip: A0 at the left, C8 at the right, **black keys at the top**
(same orientation as PianoVAM), red lines on the white-key borders, and every
**green line on a key with a finger on it** (green = keys struck in the last
100 ms; that is why `--onsets_only` is used — PianoYT's audio-derived labels
keep pedal-held keys "active"). Wrong crop, mirrored or rotated → add the id to
`data.exclude_records` in `configs/pianoyt.yaml`. Green lines on idle keys
throughout one video → probably out of sync: note the id for Phase 4.

**Send me** `results/phase0_vam_on_yt.txt`, `phase0_vam_on_vam.txt`,
`phase0_yt_on_yt.txt` and `phase0_probe_yt.txt` (the "issues" list at its end).

## Phase 1 — build the strip caches (~1–2 h, ~150 GB; can run during Phase 0)

```bash
bash scripts/build_caches.sh      # checks free disk first, then builds PianoVAM + PianoYT
tail -f logs/build_caches.log     # prints GB so far and KB per frame (~38 KB expected)
```
- Interrupted or the server rebooted → the **same command** resumes.
- Too little disk → `QUALITY=85 bash scripts/build_caches.sh` (~15% smaller).
- A re-downloaded video is detected (file size) and rebuilt automatically.
- 360p ablation cache (Phase 5), when needed:
  `CONFIGS="configs/vam_full_360.yaml" bash scripts/build_caches.sh`

## Phase 2 — train E1 and E2 in parallel (~1 day)

```bash
CONFIG=configs/vam_full.yaml bash scripts/run_experiment.sh vam_full tiled
CONFIG=configs/yt_full.yaml  bash scripts/run_experiment.sh yt_full  tiled
```
Check in the first minutes:
```bash
head -20 runs/vam_full/train.log   # must say "decoded from video: 0" (else the cache is not finished)
nvidia-smi                          # GPU busy now (was mostly idle before); ~6 GB per run (estimate)
bash scripts/status.sh              # epoch progress + latest validation
```
- After the first epoch the log prints `[epoch 0] train loss … | N min` —
  **tell me N** (estimate: 20–40 min per epoch, 25 epochs ≈ 8–17 h).
- `best.pt` is chosen by `onset_f1_best` (validation onset F1 at its best
  threshold, so miscalibrated probabilities no longer pick the wrong epoch).
- Crash / reboot → run the **same command** again (resumes from `last.pt`).
- No progress for a long time → `kill $(cat runs/vam_full/run.pid)`, then the
  same command with `train.num_workers=0` appended (the old safe mode).

## Phase 3 — evaluate: one command per model (~1 h each)

Thresholds always come from **valid**, the reported number from **test** —
`evaluate_run.sh` does both. While two trainings are running, start the
evaluations one at a time (each needs ~6–8 GB of GPU memory).

```bash
git clone https://github.com/yonghyunk1m/V2N ~/V2N     # once: V2N's released predictions

# E1  PianoVAM model on PianoVAM, and head-to-head with V2N
BASELINE_MIDI=~/V2N/predicted_midi/pianovam_test \
  bash scripts/evaluate_run.sh E1_vam_full configs/vam_full.yaml runs/vam_full/best.pt full_f1

# E2  PianoYT model on PianoYT
bash scripts/evaluate_run.sh E2_yt_full configs/yt_full.yaml runs/yt_full/best.pt onset_f1

# E3  PianoVAM model on PianoYT, zero-shot:
#     (a) with the thresholds E1 found on PianoVAM   (b) thresholds from PianoYT valid
DATA_CONFIG=configs/yt_full.yaml THRESHOLDS_FROM=results/E1_vam_full_calibrate.txt \
  bash scripts/evaluate_run.sh E3a_zeroshot configs/vam_full.yaml runs/vam_full/best.pt onset_f1
DATA_CONFIG=configs/yt_full.yaml \
  bash scripts/evaluate_run.sh E3b_zeroshot configs/vam_full.yaml runs/vam_full/best.pt onset_f1

# E4  fine-tune the PianoVAM model on PianoYT (after vam_full finished), then evaluate
CONFIG=configs/yt_finetune.yaml bash scripts/run_experiment.sh yt_finetune tiled
bash scripts/evaluate_run.sh E4_yt_finetune configs/yt_finetune.yaml runs/yt_finetune/best.pt onset_f1

# E5  PianoYT model on PianoVAM, zero-shot (a) and (b), as E3
DATA_CONFIG=configs/vam_full.yaml THRESHOLDS_FROM=results/E2_yt_full_calibrate.txt \
  bash scripts/evaluate_run.sh E5a_reverse configs/yt_full.yaml runs/yt_full/best.pt onset_f1
DATA_CONFIG=configs/vam_full.yaml \
  bash scripts/evaluate_run.sh E5b_reverse configs/yt_full.yaml runs/yt_full/best.pt onset_f1
```
Each run writes `results/<name>_test.txt` (the table: onset / onset+offset F1
at 50 and 100 ms, frame F1), `results/<name>_test.csv` (per video → paired
Wilcoxon tests, what V2N reports), `results/<name>_calibrate.txt` (the
thresholds) and `out/<name>_test/*.mid`; with `BASELINE_MIDI` also
`results/<name>_vs_baseline.txt`.

## Phase 4 — PianoYT label sync (E6)

PianoYT labels were transcribed from each video's audio. `sync_check` uses a
model as a probe and finds, per video, the label shift that maximises onset F1.
Probe with the **PianoVAM** model (trained on perfectly synced data) if E3b is
reasonable (onset F1 ≳ 0.3); otherwise use `runs/yt_finetune/best.pt`:
```bash
T=$(grep -oE 'decode\.onset_threshold=[0-9.]+ decode\.frame_threshold=[0-9.]+' results/E3b_zeroshot_calibrate.txt | tail -n 1)
echo $T      # must print the two thresholds
nohup python -m pianovam_vision.sync_check --config configs/vam_full.yaml \
    --checkpoint runs/vam_full/best.pt --data_config configs/yt_full.yaml \
    --splits train valid --csv results/sync_yt_trainvalid.csv \
    --write_offsets pianoyt_label_offsets.json $T > logs/sync_trainvalid.log 2>&1 &
nohup python -m pianovam_vision.sync_check --config configs/vam_full.yaml \
    --checkpoint runs/vam_full/best.pt --data_config configs/yt_full.yaml \
    --splits test --csv results/sync_yt_test.csv $T > logs/sync_test.log 2>&1 &
tail -f logs/sync_trainvalid.log
```
It prints the median lag (the probe's own bias, or a dataset-wide offset) and
lists the videos that deviate from it. Corrections are written for train+valid
only: the **test labels stay official**, so E2/E4 remain comparable with the
literature (the test-set lags are for the analysis/figure). Then retrain and
evaluate with the corrected labels:
```bash
CONFIG=configs/yt_full.yaml bash scripts/run_experiment.sh yt_full_sync tiled data.label_offsets=pianoyt_label_offsets.json
bash scripts/evaluate_run.sh E6_yt_sync configs/yt_full.yaml runs/yt_full_sync/best.pt onset_f1
```
The histogram of per-video lags is a figure in its own right.

## Phase 5 — ablations (one override each; ~half a day per run)

| Ablation | Command change | Question |
|---|---|---|
| data amount | old `tiled_best_v2` vs `vam_full` | already have both |
| input resolution | `configs/vam_full_360.yaml` (needs its cache) | 360p vs 720p decode |
| no augmentation | `augment.enable=false` | worth of augmentation |
| onset-only PianoYT | `train.frame_loss_weight=0` | do pedal-polluted frame labels hurt? |
| grayscale | `keyboard.grayscale=true` (same cache) | PPAN found it helps |
| inference window | `infer.window_frames=64 infer.window_hop=32` at evaluate time | model is trained on 32-frame clips |
| per-key vs global | `model.arch=strip` run, then E3 zero-shot | is transfer due to the per-key design or the tight crop? |

## Phase 6 — hand attribution (E7)

The skeleton tools (`hands.py`, `draw_hands.py`) are on `origin/main`, not on
this branch. Once E1–E4 are running: merge them, derive left/right labels per
note from PianoVAM's `Handskeleton/`, add a per-key hand head, report hand
accuracy on PianoVAM test. PianoYT has no hand labels (MediaPipe on PianoYT
would give pseudo-labels, if needed).

## Results tables to fill

**A — PianoVAM test (9 videos).** Onset P/R/F1 @50 and @100 ms,
onset+offset F1 with the V2N rule (`offset within τ`) and the mir_eval default,
frame F1. Rows: S2S, V2R, PPAN, V2N (V2N's Table 1), old `tiled_best_v2`,
`vam_full`, ablations.

**B — PianoYT test (20 videos; 163 is gone from YouTube).** Onset F1 @50 and
@100 ms. Rows: S2S 64, V2R 64, PPAN 68 (@100, R3-pretrained), E2, E4, E6, and
E3 (zero-shot).

**C — cross-dataset matrix.** Train {PianoVAM, PianoYT, VAM→YT} × test
{PianoVAM, PianoYT}.

## Gotchas

- Run everything from `tivit_pianoyt/vision_transcription`; `--flags` before
  `key=value` overrides.
- `evaluate` / `calibrate` / `infer` now score **whole videos** by default;
  `--max_frames N` only for quick checks.
- Train log says `decoded from video: N` with N>0 → cache incomplete or stale
  (the reason is printed) → run `build_cache` again.
- `data.label_offsets` shifts labels everywhere (training, evaluation,
  previews) — keep separate run names for with/without.
- The old configs (`tiled_best.yaml`, `pianoyt.yaml`) behave exactly as before;
  the paper runs use `vam_full` / `yt_full` / `yt_finetune` / `vam_full_360`.
- PianoVAM: our split and (corrected, March-2026) labels equal V2N's v1.1. We
  exclude one "corrupt" training video that V2N trains on,
  `2024-09-05_21-37-08`. The official file is 185483494 bytes, sha256
  `40ca0af6e396714ec1b8a1ab70072af6b50d33959b1f42dc2cc24e230f931300`. Compare
  with `ls -l` / `sha256sum` on your copy: if it differs, re-download
  `Video/2024-09-05_21-37-08.mp4` from the Hugging Face dataset
  (PianoVAM/PianoVAM_v1), check it with `check_data --probe`, and drop it from
  `exclude_records` to train on all 81 like V2N.
