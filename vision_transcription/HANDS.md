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
| `pianovam_vision/hand_notes.py` | **fast path (§5)**: the skeleton-only hand model trained without video + a note-sequence model; folds (`official`, `loro`, `lopo`, `session`), `cv`, `robust`, `e2e` |
| `pianovam_vision/hand_report.py` | all tables from the per-note outputs: accuracy with block-bootstrap CIs, recording-level tests, error analysis by subset, precision–coverage, confusion |
| `hands.py` (added) | `ayf_scores` (At Your Fingertips' rule), `decide(joint=True)`, `FingerPrior` + `viterbi_fingers` (fingering HMM per hand) |
| `evaluate_hands.py` (added) | method `ayf`; `--save_probs` (per-note probabilities/scores for `hand_report`) |
| `configs/hand_notes.yaml` | the fast path's recipe |
| `scripts/preflight_hands.sh`, `hands_suite.sh`, `jobs_hands_suite.txt`, `hands_reports.sh` | readiness check, every experiment in one restartable background queue, all reports |
| `tests/test_hand_notes.py` | 17 tests (rules, prior + Viterbi, note data, the 4 feature variants train and predict, residual start, folds never leak, perturbations, end-to-end counts, report) |
| `scripts/extract_landmarks.py` | MediaPipe Hands landmarks for any video in PianoVAM's `Handskeleton` JSON layout (`extract`, parallel, restartable), agreement with a reference (`compare`), and which settings reproduce PianoVAM's own landmarks (`validate`). Own environment (`~/venvs/mediapipe`, mediapipe 0.10.14); imports nothing from the project |
| `tests/test_extract_landmarks.py` | 3 tests (duplicate hand tags, the JSON is read by `hands.read_skeleton_json`, compare: identical / shifted / swapped) |

Transcription with a hand model is bit-identical to the checkpoint it was built
on (tested: same weights with and without the branch give equal onset/frame
outputs; a frozen run's valid onset F1 does not move).

## 4. Results so far

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

**The learned model (GPU runs of 2026-11-14, one seed each).** 450 unseen manual
notes (valid 150 + test 300), hand / hand + finger %:

| method | hand | hand + finger |
|---|---|---|
| pitch split | 76.4 | – |
| classic rule / key-ROI rule | 97.6 / 97.8 | 82.0 / 82.4 |
| dataset labels (87.6 % decided) | 86.2 | 81.8 |
| **vam_hands (video + landmarks, transcription frozen)** | **99.6** | **91.8** |
| landmarks only (`vam_hands_skel`) | 99.8 | 91.8 |
| video only (`vam_hands_video`) | 99.8 | 85.8 |
| joint fine-tuning (`vam_hands_joint`) | 99.1 | 88.7 |

Automatic test labels (36,443 notes): fusion 98.9 / 97.1 vs key-ROI rule 99.2 / 91.8.
Joint fine-tuning made the notes worse (93.65 / 97.16 vs 94.20 / 97.65 for E1 run 2)
and is dropped; chord matching was slightly worse for the model and stays off.
Read per note (audit of 2026-11-14): on the 426 manual notes matched to the model's
own notes, model 92.5 % vs key-ROI rule 82.9 % (block-bootstrap 95 % CI of the
difference +6.9 to +12.5); against the dataset labels the gain is coverage (94.9 vs
93.8 % where they answer, 76 % on the 54 they leave out). The video adds nothing
measurable to the finger once the landmarks are there, so the learned part can be
studied with the landmarks alone — that is the fast path of §5.

**Cross-validation suite (§5, run 2026-11-14/15 on the RTX 3090: 160 trainings,
20 h, no failure; reports in `results/hn/`).** Hand / hand + finger % over all
1,800 manual notes, the evaluated recording never in training; `keyrel:seq`
(3 seeds, s.d. ≤ 0.3) unless noted:

| test | ours | At Your Fingertips' rule | MIDI only (`none`) | raw image coords |
|---|---|---|---|---|
| unseen recording (`loro`) | **99.3 / 93.0** | 98.3 / 86.1 | 89.8 / 55.1 | 98.5 / 84.8 |
| unseen pianist (`lopo`) | **99.5 / 92.2** | 98.3 / 86.1 | 88.2 / 51.0 | 98.1 / 78.6 |
| unseen camera set-up (`session`) | **99.6 / 90.6** | 98.3 / 86.1 | 90.1 / 52.8 | 96.3 / 59.4 |

* vs the rules: +6.9 points hand + finger over At Your Fingertips (CI +5.5 … +8.4),
  better than the key-ROI rule on 11 / 11 recordings (sign-flip p = 0.001).
* What matters is the key-relative input, not the architecture: on the official
  split, no sequence head / no key context / no time context / hand-only training
  all land within 1 point (none significant); `seq`, `frame` and `+hmm` decoders
  are equal on clean input (93.0 / 93.3 / 93.2–93.5). Training plateaus at
  1,500–2,000 of the 3,000 steps.
* Robustness (`loro`, seed 1, frame read-out vs key-ROI rule): landmark noise of
  ¼ key 93.2 vs 85.4; hand tags swapped on 20 % of frames 91.2 vs 68.3; hands
  missing on 50 % of frames 91.5 (hand 99.3) vs 78.0. **Keyboard corners off by
  ¼ / ½ / 1 white key (0.5 / 1 / 2 % of the width) break every method**: 55 / 32 / 25
  vs rule 56 / 32 / 28. Trained with corner jitter (0.005) the sequence decoder
  keeps 89.9 / 85.8 / 60.9 (hand 99.4 / 98.8 / 92.3) at no clean cost on the session
  folds (90.9; one seed) — the sequence head is what turns the jitter into
  robustness (the same model's frame read-out: 73.6 / 58.9 / 41.0). → round 2.
* End to end (test, 9 recordings, F1 at 100 ms, automatic labels): KeyWise notes
  97.8 → + hand 96.7 → + finger 94.8; V2N notes 97.1 / 96.1 / 94.4; hands from MIDI
  only 90.6. On the 300 manual test notes no KeyWise note got the wrong hand.
* For the score use the hand of the frame read-out: the sequence head moves notes
  to the other hand when one hand is not detected (10 of its 11 extra hand errors,
  one passage of 2024-02-15_20-07-54; with 50 % of hands missing its hand accuracy
  is 92.8 vs 99.3).
* Where the remaining errors are: 76 % of the wrong notes are notes whose automatic
  label is missing (42 %, the 12 % "Noinfo" notes, model 74 %) or wrong (34 %);
  where the dataset gives a label the model equals it (95.5 vs 95.7). Finger errors
  go one finger toward the thumb at the outer registers only (key-ROI rule: right
  hand above C5 82 vs 4, left hand below C3 30 vs 0; balanced in the middle; differs
  per camera set-up): perspective of fingertips above the key plane. The dataset's
  labels share it. Correcting the output afterwards gains only 0.4–0.7.
* **Round 2** (`loro`, all 1,800 manual notes; hand + finger %, sequence decoder;
  clean and perturbations seed 1):

  | trained with | clean | corners off ¼ / ½ / 1 key | 50 % hands missing | 20 % tags swapped |
  |---|---|---|---|---|
  | nothing extra (`keyrel`) | 92.9 | 56.2 / 32.4 / 25.9 | 85.6 | 90.6 |
  | corner jitter 0.0025 (⅛ key) | 92.6 | 89.2 / 65.4 / 44.1 | 88.9 | 89.6 |
  | corner jitter 0.005 (¼ key) | 91.7 | 90.7 / 85.2 / 64.2 | 86.3 | 87.4 |
  | corner jitter 0.01 (½ key) | 88.9 | 88.4 / 87.2 / 79.9 | 79.3 | 84.8 |
  | manual labels (10 % of windows) | 93.1 | 55.7 / 32.8 / 27.7 | 88.9 | 90.1 |
  | jitter 0.005 + manual labels | 92.1 | 91.3 / 86.6 / 63.9 | 85.2 | 88.0 |
  | key-ROI rule | 85.9 | 56.2 / 31.7 / 28.2 | 78.0 | 68.3 |

  - Corner jitter: each level holds up to about its own size of error, then falls.
    0.005 costs ~1 point on clean corners (3 seeds 92.2 vs 93.0, CI −0.1 … +1.7,
    6 / 5 recordings, n.s.; LOPO −1.4, one seed; official −2.1 on 450 notes) and keeps
    the hand at 99.3 / 99.1 / 91.4 % with corners ¼ / ½ / 1 key off (no jitter:
    95.9 / 89.6 / 79.7). With jitter only the sequence decoder is usable (frame
    read-out 86.6 clean). It also removes the sequence head's hand flips when a hand
    is missing (hand 98.7 vs 92.8 with 50 % of hands missing).
  - Manual labels of the other recordings: **no gain** (93.1 vs 93.0, 3 seeds,
    recordings 4 / 4; 25 % of windows 93.6, one seed, n.s.; LOPO 92.3 vs 92.2), although
    every fold used 1,650 notes, each seen over 100 times (the labelled notes are the
    first 150 of each recording, so a manual window is mostly manual notes). So the
    remaining ~7 % errors come from the landmarks, not from noisy training labels:
    ring / little finger (R4→R3 22, R5→R4 21 notes), Noinfo notes (74 %), one hand
    seen (63 %). The automatic labels fail on the same notes for the same reason.
  - Two settings for the paper: no jitter when the corners are accurate (PianoVAM's
    are set by hand), jitter 0.005 when they come from a detector or a crop box.
    The PianoYT test decides the level.

## 5. The experiments for the second paper (fast path, cross-validation)

Three test recordings and one unseen pianist cannot carry a paper. The fast path
(`hand_notes.py`) trains the landmarks-only model in minutes (no video decoding),
so every claim is tested by cross-validation over **all 1,800 manual notes**
(11 recordings, 7 pianists, 3 recording sessions). Manual labels are never used
for training or model selection anywhere; folds train for a fixed number of steps.

**Model.** Frame part = `HandBranch` with landmarks only (as `vam_hands_skel`),
run on a 12-frame clip around each onset and read on the note's key; note part =
a BiGRU over the notes in playing order (hidden feature of the frame part + pitch
and timing context), whose output is added to the frame read-out (starts at it,
learns corrections). Decoders compared on the same notes: `frame` (frame read-out
= what vam_hands_skel does), `seq` (note-sequence model), `+hmm` (Viterbi per hand
with a finger-transition prior learned from the training labels).
Hard chord constraints were rejected on the data: in the manual labels 22 of 306
one-hand chords break "fingers in pitch order", 13 because one finger plays two keys.

**Folds** (`hand_notes folds`): `official` (train + ext-train → valid + test),
`loro` (leave one manually labelled recording out: 11), `lopo` (leave one pianist
out — all their recordings leave training: 7), `session` (leave one session out:
Feb / Mar–Apr / Sep 2024; the camera moved between them — keyboard at x 121–1839 px
in Feb, 251–1652 px in Sep, 18 % narrower).

**Experiments** (`scripts/jobs_hands_suite.txt`, in priority order) and the
question each answers:

| runs | question |
|---|---|
| `official_keyrel` ×3 seeds | does the fast path reproduce vam_hands_skel (99.8 / 91.8 on the 450 notes)? |
| `loro_keyrel`, `lopo_keyrel` ×3 | does the learned model beat the rules (classic, key-ROI, At Your Fingertips) and match the dataset labels on all 1,800 notes, for unseen recordings and unseen pianists? |
| `seq` vs `frame` vs `+hmm` (every run) | does note-sequence decoding help, and where (ring / little finger, chords, fast passages, Noinfo notes)? |
| `*_none` | a learned MIDI-only hand separation (pitch + timing, no video): what does vision add for the hand? |
| `*_image`, `*_strip` | is the key-relative representation what generalises (raw image or absolute strip coordinates instead)? Decisive in `session` (unseen camera set-up). |
| `session_*`, `lopo_*`, `robust` | Direction 3: unseen pianists and camera set-ups; corrupted keyboard corners (0.5–2 % of the width), landmark noise, swapped hand tags, missed hands |
| `*_handsup` | do the fingers help the hand? Same model trained on the hand only |
| `official_noseq / nokeyctx / notimectx` | which context matters |
| `*_cornerjit` | does corner jitter in training buy robustness to keyboard-detection errors? |
| `e2e` on `official_keyrel` | Direction 2: notes + hand + finger from video only, KeyWise vs V2N notes, per recording, errors split into missed note / wrong hand / wrong finger; hand-separated MIDI per recording |

**Statistics** (`hand_report`): accuracy over all notes (no decision = wrong);
mean ± s.d. over seeds; 95 % CI by block bootstrap (blocks of 10 consecutive notes,
within recordings) and, for differences, also with recordings resampled; exact
sign-flip test over recordings; McNemar only as a note-level aside;
precision–coverage (to compare with the dataset labels' 88 % coverage and with
At Your Fingertips' P/R/F1).

**At Your Fingertips' rule** (`hands.ayf_scores`), re-implemented from Sec. III-E:
Gaussian of each fingertip's horizontal position around the key centre (σ = key
width), normalised over the detected fingers, press frames weighted 0.5ⁿ from the
first, arg max over the 10 fingers. The paper gives neither the decay nor the
fingertip-to-position mapping; these are our choices. On the 300 manual test notes:
89.0 % hand + finger (key-ROI rule 89.3, classic 88.3).

## 6. How to run (GPU server)

**The second-paper experiments (§5):**

```bash
cd ~/tivit_vam_v2_repo/vision_transcription && git pull
bash scripts/preflight_hands.sh        # GPU, code, tests, data, speed -> READY / NOT READY
BASELINE_MIDI=/home/achatzigiannis/V2N/predicted_midi/pianovam_test bash scripts/hands_suite.sh
tail -f logs/hands_suite.log           # progress; stop: kill $(cat logs/hands_suite.pid)
bash scripts/hands_reports.sh          # all reports so far -> results/hn/*.txt (any time)
```

One fold = 3,000 steps; `preflight_hands.sh` measures the speed and prints the
total (≈160 training runs).

**Round 2** (`scripts/jobs_hands_round2.txt`, ~146 runs ≈ 21 h, most important
first): corner jitter on all 11 recordings with 3 seeds and at three levels, and
training that also uses the manual labels of the *other* recordings
(`hand_notes.manual_frac`, never the evaluated one's):

```bash
ROBUST_RUNS="loro_cornerjit loro_manual loro_cornerjit0025 loro_cornerjit01 loro_cj_manual" \
  bash scripts/hands_suite.sh scripts/jobs_hands_round2.txt
bash scripts/hands_reports.sh          # adds results/hn/round2_*.txt and robust2_*.txt
```

Single pieces by hand:

```bash
python -m pianovam_vision.hand_notes folds --config configs/hand_notes.yaml --scheme loro,lopo,session
python -m pianovam_vision.hand_notes cv --config configs/hand_notes.yaml --scheme loro --out runs/hn/loro_keyrel --seeds 1,2,3
python -m pianovam_vision.hand_report --run keyrel=runs/hn/loro_keyrel --run midi_only=runs/hn/loro_none --out results/hn/loro_manual.txt
python -m pianovam_vision.hand_notes e2e --config configs/hand_notes.yaml --model runs/hn/official_keyrel/official_s1/model.pt \
  --pred KeyWise=out/E1s2_vam_full_test --pred V2N=$BASELINE_MIDI --split test --out out/hn/e2e
```

**The video models of §4 (done):**

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

## 7. Open points

* More manual labels (human work): ~150 notes in each of the 7 test recordings
  without them would double the unseen test set (PianoVAM's annotation tool is public).
* Other cameras / keyboards beyond PianoVAM's three sessions: no public video set
  has manual hand / finger labels per note (At Your Fingertips' APFD is labelled by
  their own rule; PianoMotion10M has 3-D hands but no fingers). Plan: PianoYT test
  videos (already in this repo's pipeline, crop box = rough corners) + MediaPipe
  landmarks + a small manual set (~1,000 notes hand, ~300 finger) → zero-shot test.
* For a new video the landmarks are extracted first with `scripts/extract_landmarks.py`
  (MediaPipe Hands, 21 points, the same JSON layout and tag convention). MediaPipe's
  default settings (video mode, model complexity 1, confidences 0.5) **reproduce
  PianoVAM's own landmarks**: on the first 20 s of 2024-02-17_21-44-37,
  2024-03-11_22-23-29 and 2024-09-02_14-10-41 the median difference is 0.1–0.2 px,
  the same hands are found in 99–100 % of the frames, no tags are swapped and the
  frames are in sync; complexity 0 or static-image mode differ by 2–6 px. So new
  data gets landmarks like the ones the model was trained on.

  ```bash
  PY=~/venvs/mediapipe/bin/python      # python3 -m venv ~/venvs/mediapipe; pip install mediapipe==0.10.14
  $PY scripts/extract_landmarks.py validate --root /raid_storage/data_achatzigiannis/PianoVAM_v1.0
  $PY scripts/extract_landmarks.py extract <videos or globs> --out_dir <dir> --workers 16
  ```
* Candidate test sets (checked 2026-11-16): **SKY-Piano** (ISMIR 2026, Hugging Face
  `SKY-Piano/SKY-Piano`, 51.7 GB, gated; top-view video, MIDI, fingering from motion
  capture — the first labels not made from MediaPipe) and **R3** (PPAN's dataset,
  81.4 GB raw on the Internet Archive; silent-piano MIDI, no hand labels).
* Towards a playable score (PhD goal): the two-track MIDI is the hand split a
  notation program needs (one staff per hand). Still to do: rhythm quantisation /
  beat tracking from the performance MIDI, voices within a hand, and MusicXML export
  with the finger numbers (MIDI has no field for them).
* Not done: a wider strip with the wrists (new cache), using hands to improve
  transcription.
