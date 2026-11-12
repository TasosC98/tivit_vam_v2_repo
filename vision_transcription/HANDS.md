# KeyWise-hands — which hand, and which finger, played each note

Branch `KeyWise-hands` (from `paper1-review`, e287984). PianoVAM only: PianoYT has
no hand or finger labels. The transcription network and all paper results are
untouched; the hand model is an extra branch on top of it.

For every transcribed note the output says **left or right hand** and **which
finger** (1 = thumb … 5 = little finger), with a confidence:

```
=== 2024-02-17_21-44-37: 530 notes ... ===
  left hand     190 (35.8%)   right hand    340 (64.2%)
   time (s)   note  hand   finger      p(hand)
     14.400    C#5  right  2 (index)      ...
     15.534    C#3  left   5 (little)     ...
```

plus a two-track MIDI (Left hand / Right hand), a TSV, a piano roll coloured by
hand, and a sheet of video frames with each pressed key outlined in its hand's
colour and the finger written under it (`transcribe_hands.py`).

---

## 1. Data (checked 2026-11-12)

PianoVAM **v1.2** (Hugging Face `PianoVAM/PianoVAM_v1`; the old name
`PianoVAM_v1.0` redirects) added two label folders. Our dataset copies did not
have them; `scripts/download_hand_data.py` fetches them (23 MB, no login).

| folder | what | size |
|---|---|---|
| `Fingering/` | automatic hand (L/R/Noinfo) + finger (1–5) for every note | 106 recordings, 525,483 notes |
| `Fingering_GT/` | **manual** labels, the only ground truth | 1,800 notes: first 150 notes of 10 recordings, first 300 of one |
| `Handskeleton/` | MediaPipe Hands, 21 landmarks × 2 hands per native frame (60 fps) | 108 files, 12.1 GB (we already had it) |

What `check_hands` verified on all 106 recordings (local copy, identical to Hugging Face):

* `Fingering/` rows equal our (corrected) `TSV/` row for row → the labels belong
  to exactly our notes. 19.9 % of the notes are `Noinfo` (no finger qualified);
  labelled notes: train 80.3 %, ext-train 75.2 %, valid 78.8 %, test 86.3 % —
  ≈ 413,000 labelled training notes.
* The manual labels line up with the first rows. We reproduce the dataset
  README exactly: the automatic labels cover 88.2 % of the 1,800 manual notes,
  and where they answer they have the right hand for 99.2 % and the right hand
  and finger for 95.7 %.
* **The Handskeleton "Left"/"Right" tags are mirrored** (MediaPipe's selfie
  convention). Nearest fingertip with the tags as stored: 2.1 % right hand on the
  manual labels; swapped: 97.9 %. Swapped, the nearest-fingertip hand agrees with
  the dataset's labels for 98.7 % of the notes (worst recording 94.2 %)
  (`hands.swap_tags: true`, checked by `check_hands --skeleton`). The old
  `hands.py` of June (commit 76767e7) used the tags as stored, i.e. backwards.
* The landmarks project correctly into the strip: at the onset the true finger's
  tip is at 0.44 of the key length on black keys (they end at 0.62) and at 0.77
  on white keys; each hand is detected in ~97 % of the frames (a few short pieces
  are played with one hand).
* **The landmarks of the Sep 04–05 videos are out of sync.** v1.1 replaced those
  videos with sync-corrected versions, but `Handskeleton/` was not redone: for 18
  of them the skeleton has 35–42 frames (1.2–1.4 s) more than the video, and the
  fingertips reach the pressed keys 2–5 frames (67–167 ms) too early in the
  landmarks (one recording: 4 frames too late), the same all through the
  recording. `check_hands --skeleton --write_offsets` finds this without labels
  (the shift at which fingertips sit on the pressed keys; in-sync recordings all
  agree) and wrote `configs/pianovam_skeleton_offsets.json`: 12 recordings get an
  offset (11 of the 18, one train recording by 2 frames), all train/ext-train, so
  valid/test are unaffected. On the manual labels it helps (key-ROI rule
  hand+finger 85.2 → 85.9 %, from the one manual recording affected), while
  agreement with the automatic labels drops slightly (91.2 → 90.6 %) — the
  dataset's own labels for those recordings were made with the shifted landmarks
  (that recording has the lowest coverage, 71 %, and finger accuracy, 88 %, of all
  11). Training targets from those recordings are therefore a little noisier.

**Caveat for evaluation.** Of the 11 manually labelled recordings, 8 are in
train/ext-train. A model trained on train + ext-train has seen those videos (with
the automatic labels), so a learned model is scored on the 3 unseen ones only:
valid `2024-02-15_20-07-54` and test `2024-02-17_21-44-37`, `2024-09-02_14-10-41`
(450 notes; 300 in test). Rules without training can use all 1,800. The
automatic labels of the whole test split (36,443 labelled notes) give a large
second set, with the caveat that they come from MediaPipe themselves.

## 2. Why key ROIs, and how they beat a whole-keyboard view here

The concern: KeyWise looks at the keyboard through one narrow ROI per key (the
CNN sees about 3 white keys), while V2N looks at the whole keyboard (800×144 plus
a bottom margin with the wrists). Telling the left hand from the right, or the
index from the middle finger, needs the whole hand.

V2N itself has **no hand or finger output** (checked in arXiv 2608.03419: onset,
offset, key-hold and velocity heads only; future work lists localisation, sync,
pedal and audio-visual fusion, not hands). The comparison that matters is with
the dataset's own fingering pipeline (same authors) and the usual MediaPipe rule,
and end-to-end: our notes + hands against V2N's notes + the same hand decision.

What the per-key design gives us that a global feature does not:

1. **The decision is per key, where it is needed.** A note is (key, onset frame);
   the hand/finger branch outputs 10 classes for every key and frame, and a note
   reads them at its own key and frames — exactly how the velocity head works
   (which beat V2N on ++Vel). A global vector would first have to work out which
   part of its feature belongs to the note.
2. **The skeleton meets the tiles in key coordinates.** The homography that cuts
   the ROIs also maps every MediaPipe landmark into the strip, so for each key we
   compute where each fingertip is *relative to that key*, with the real key shapes
   (black keys are shorter and narrower). This turns "which finger pressed C#4"
   into a local, geometric question. The key-ROI rule alone already beats the
   classic rule (nearest fingertip to the key centre) — see §4.
3. **Context across keys fixes the narrow ROI.** The hand branch adds residual
   dilated convolutions along the key axis (±30 keys, about a hand span each way)
   and one along time, and a per-key embedding (pitch prior). The ROI decides
   *where*; the context decides *which hand*.
4. **One finger per key at a time.** Optional chord matching: the notes of one
   chord get different fingers (Hungarian algorithm).
5. **Small.** ~0.27 M parameters on top of KeyWise's 1.30 M (V2N: 28.2 M).

Wrists and forearms lie in front of the keyboard quad, outside our strip (see the
frame sheet: the strip is the keys only). The landmarks carry the wrist, so the
fusion model has it; the video-only ablation does not — that ablation tells
whether a bottom margin (a new strip cache) would be worth building.

## 3. What was added

| file | what |
|---|---|
| `pianovam_vision/hands.py` | labels (`read_fingering`, `build_hand_roll`), skeleton reading + mirrored-tag fix + projection into the strip (`load_skeleton`, cached), real key layout, rules (`pitch_split`, `fingertip_scores` tip/roi), model read-out (`probs_scores`), `decide` (+ chord matching), TSV / two-track MIDI writers |
| `pianovam_vision/model.py` | `SkeletonKeyFeatures` (landmarks → 84 features per key), `HandBranch`; `TiledVisualOnsetsFrames(frames, skel=None)` adds `hand_logits` (B,T,88,10) when `model.hands.enable` |
| `pianovam_vision/dataset.py` | `hand` targets and `skel` input per clip (landmarks moved with the strip augmentation; one hand hidden in 10 % of clips) — only for hand configs |
| `pianovam_vision/train.py` | hand loss (cross-entropy on labelled key-frames), `train.freeze_transcription` (weights **and** BatchNorm statistics fixed), valid `hand_acc` / `finger_acc`, `select_metric: finger_acc` |
| `pianovam_vision/infer.py` | `predict_outputs` (also hand probabilities); `predict_rolls` unchanged |
| `pianovam_vision/check_hands.py` | data check + skeleton cache |
| `pianovam_vision/evaluate_hands.py` | all methods, given-notes and end-to-end, manual and automatic labels, McNemar tests |
| `pianovam_vision/transcribe_hands.py` | the decision display (TSV, MIDI, piano roll, frame sheet, printed table) |
| `scripts/download_hand_data.py`, `scripts/evaluate_hands.sh` | data download; full scoring of a run |
| `scripts/queue_hands.sh` + `scripts/jobs_hands.txt` | the four runs trained and scored one after another (restartable) |
| `configs/vam_hands.yaml` (+ `_video`, `_skel`, `_joint`) | fusion model on a frozen E1 run 2; ablations; joint fine-tuning |
| `configs/pianovam_skeleton_offsets.json` | landmark time offsets per recording (written by `check_hands --write_offsets`) |
| `tests/test_hands.py` | 15 tests (labels, geometry, tag swap, rules, landmark alignment, chord matching, frozen training, transcription unchanged) |

Transcription with a hand model is bit-identical to the checkpoint it was built
on (tested: same weights with and without the branch give equal onset/frame
outputs; a frozen run's valid onset F1 does not move).

## 4. Results so far (no training needed)

Measured on the edit PC with `evaluate_hands` (landmark offsets on, read-out
window 8 frames, chosen on train recordings only). Accuracy is over **all**
notes: a note without a decision counts as wrong.

**Given notes, manual labels, all 11 recordings (1,800 notes)** — rules have no
training, so all 11 are fair for them:

| method | decided | hand | hand + finger |
|---|---|---|---|
| pitch split at C4 (no vision) | 100 | 81.3 | – |
| nearest fingertip to the key centre (classic) | 100 | 98.1 | 84.4 |
| **nearest fingertip to the key ROI (ours)** | 100 | **98.6** | **85.9** |
| dataset's automatic labels (`Fingering/`) | 88.2 | 87.6 | 84.4 |

Exact McNemar: key ROI vs classic, hand p = 0.053, hand+finger p = 0.018;
key ROI vs dataset labels, hand p < 0.0001 (they leave 11.8 % undecided),
hand+finger p = 0.084. Of the key-ROI rule's errors, 3/4 are a neighbouring
finger; wrong hands are 1.4 %. On the 36,443 automatically labelled test notes:
key ROI 99.2 / 91.8 vs classic 98.8 / 88.3 (p < 0.0001 both).

**Where the learned model has to win:** the finger (hand is ~solved by geometry).
Landmark tips are noisy by about one finger; the video shows which key actually
goes down.

**End-to-end on test** (transcribed notes, onset within 50 / 100 ms; hands by
the key-ROI rule for every note source; KeyWise = the three E1 runs; V2N = its
released MIDI). Note+Hand F1 can never exceed Note F1.

| notes from | Note F1 | Note+Hand F1 | Note+Hand+Finger F1 |
|---|---|---|---|
| *all test notes (automatic labels, 36,443)* | | | |
| KeyWise run 1 / 2 / 3 | 96.6 / 95.6 / 97.3 · 98.7 / 98.5 / 98.7 | 95.7 / 94.7 / 96.4 · 97.8 / 97.6 / 97.8 | 88.6 / 87.7 / 89.3 · 90.5 / 90.3 / 90.5 |
| V2N | 95.4 · 97.6 | 94.5 · 96.7 | 87.6 · 89.7 |
| *first 150 notes of the 2 manual test recordings (300)* | | | |
| KeyWise run 1 / 2 / 3 | 86.9 / 86.0 / 90.8 · 96.9 / 96.2 / 97.3 | same as Note | 77.0 / 76.8 / 80.6 · 86.9 / 86.0 / 87.1 |
| V2N | 95.2 · 96.9 | same as Note | 86.4 · 87.8 |

(50 ms · 100 ms.) Over the whole test split our notes carry the hands better
than V2N's in every run. On the 300 manual notes V2N is ahead at 50 ms: these two
recordings are two of the three where KeyWise's onsets sit late (the timing
issue of Paper 1); at 100 ms they are level. Every hand was right on these 300
notes for both systems; the gap is purely the notes.

Chord matching (one finger per chord note) did not help the rules (manual test
89.3 → 89.3, automatic 91.8 → 91.0 hand+finger); it stays off unless it helps the
model.

The learned KeyWise-hands numbers come from the GPU run (§5). Only the 450
unseen manual notes (valid + test) and the automatic test labels may be used
for it.

## 5. How to run (GPU server)

```bash
cd ~/tivit_vam_v2_repo/vision_transcription && git fetch && git checkout KeyWise-hands
# 0. data, once (Handskeleton/ is usually there already; add --skeleton if not)
python scripts/download_hand_data.py --root /raid_storage/data_achatzigiannis/PianoVAM_v1.0
python -m pianovam_vision.check_hands --config configs/vam_hands.yaml --skeleton --workers 4
#    -> builds data.skeleton_cache (~0.4 GB), lists the 12 recordings whose landmarks are
#       out of sync (already corrected by configs/pianovam_skeleton_offsets.json; add
#       --write_offsets <file> only to regenerate it), ends with "VERDICT: hand data complete"
# 1+2. train and score all four runs one after another (scripts/jobs_hands.txt):
BASELINE_MIDI=/home/achatzigiannis/V2N/predicted_midi/pianovam_test \
  bash scripts/queue_hands.sh scripts/jobs_hands.txt
#    or one run by hand:
#    CONFIG=configs/vam_hands.yaml bash scripts/run_experiment.sh vam_hands tiled
#    BASELINE_MIDI=... bash scripts/evaluate_hands.sh vam_hands configs/vam_hands.yaml runs/vam_hands/best.pt
# 3. any recording, with the decision display
python -m pianovam_vision.transcribe_hands --config configs/vam_hands.yaml \
  --checkpoint runs/vam_hands/best.pt --record_time 2024-09-02_14-10-41 \
  --out_dir out/hands_demo decode.onset_threshold=0.90 decode.frame_threshold=0.70
```

`runs/vam_full_s2/best.pt` must exist (E1 run 2; a copy is in
`PHD/server/runs/vam_full_s2`). Use the onset/frame thresholds that
`results/E1s2_vam_full_calibrate.txt` gives (evaluate_hands.sh reads them).

## 6. Open points

* The learned model needs the GPU run before any claim; report the three unseen
  manual recordings (450 notes) **and** the automatic test labels, with McNemar
  tests against the key-ROI rule and the dataset labels.
* 300 manual test notes are few: a 1-point difference is 3 notes. More manual
  labels on test recordings would make the comparison much stronger.
* For a new video (not PianoVAM) the landmarks must be extracted first (MediaPipe
  Hands, 21 points, same JSON layout); the video-only model needs nothing extra.
* Not done: a wider strip with the wrists (new cache), finger crossing / thumb-under
  constraints, using hands to improve transcription.
