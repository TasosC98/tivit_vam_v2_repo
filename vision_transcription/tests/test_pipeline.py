"""Tests for labels, decoding, config, caching and metrics (no GPU/video needed).

    cd vision_transcription && python -m pytest tests/ -q

Tests needing torch / OpenCV / mir_eval are skipped where those are missing
(e.g. the Windows edit box).
"""
import json

import numpy as np
import pytest

from pianovam_vision.config import load_config, load_yaml
from pianovam_vision.labels import Note, build_target_rolls, read_reference, shift_notes
from pianovam_vision.decode import decode_notes
from pianovam_vision.metadata import Recording, load_recordings_pianoyt
from pianovam_vision.video import plan_frame_sampling


def test_target_roll_shapes_and_content():
    fps = 30.0
    notes = [Note(onset=1.0, offset=1.5, pitch=60, velocity=100)]
    n = int(3 * fps)
    frame, onset, vel = build_target_rolls(notes, n, fps, onset_window_frames=2)
    assert frame.shape == (n, 88) and onset.shape == (n, 88)
    k = 60 - 21
    # held from frame 30..45
    assert frame[30, k] == 1.0 and frame[44, k] == 1.0 and frame[46, k] == 0.0
    # onset region is 2 frames
    assert onset[30, k] == 1.0 and onset[31, k] == 1.0 and onset[32, k] == 0.0
    assert abs(vel[30, k] - 100 / 127) < 1e-6


def test_notes_after_window_are_dropped_not_stacked_on_last_frame():
    # A capped video (train.max_frames_per_record) covers only the start of the
    # labels; later notes must not all become onsets on the last covered frame.
    fps, n = 30.0, 90                                    # 3 s window (frames 0..89)
    notes = [Note(1.0, 1.5, 60, 80), Note(2.97, 3.4, 62, 80),    # inside / on frame 89
             Note(10.0, 10.5, 64, 80), Note(20.0, 21.0, 65, 80)]  # far outside
    frame, onset, _ = build_target_rolls(notes, n, fps, onset_window_frames=2)
    assert onset[-1].sum() == 1 and onset[-1, 62 - 21] == 1      # only the real onset
    assert frame[:, 64 - 21].sum() == 0 and frame[:, 65 - 21].sum() == 0
    assert frame[-1, 62 - 21] == 1                                # cut at window end


def test_decode_recovers_notes():
    fps = 30.0
    notes = [
        Note(1.0, 1.5, 60, 80),
        Note(1.0, 2.0, 64, 80),
        Note(2.5, 3.0, 67, 80),
    ]
    n = int(4 * fps)
    frame, onset, _ = build_target_rolls(notes, n, fps, onset_window_frames=2)
    # Treat the binary targets as "perfect" probabilities.
    est = decode_notes(onset, frame, fps, onset_threshold=0.5,
                       frame_threshold=0.5, min_duration_s=0.03)
    got = sorted((round(e.onset, 2), e.pitch) for e in est)
    want = sorted((round(x.onset, 2), x.pitch) for x in notes)
    assert got == want
    # offsets within one frame (1/fps)
    for e in est:
        ref = next(x for x in notes if x.pitch == e.pitch and abs(x.onset - e.onset) < 0.05)
        assert abs(e.offset - ref.offset) <= 1.0 / fps + 1e-6


def test_retrigger_same_pitch():
    fps = 30.0
    notes = [Note(1.0, 1.3, 60, 80), Note(1.4, 1.8, 60, 80)]
    n = int(3 * fps)
    frame, onset, _ = build_target_rolls(notes, n, fps, onset_window_frames=2)
    est = [e for e in decode_notes(onset, frame, fps) if e.pitch == 60]
    assert len(est) == 2


def test_pianoyt_csv_corners_and_splits(tmp_path):
    # id, url, split(1=train/3=test), min_y, max_y, min_x, max_x
    csv = tmp_path / "pianoyt.csv"
    csv.write_text(
        "100,'https://y/watch?v=a',1,666,999,45,1874\n"
        "101,'https://y/watch?v=b',3,663,1005,39,1867\n"
        "102,'https://y/watch?v=c',1,100,200,10,300\n",
        encoding="utf-8",
    )
    recs = load_recordings_pianoyt(csv, valid_frac=0.0)
    by_id = {r.record_time: r for r in recs}

    # crop box -> corners in order LT, RT, RB, LB (min_x/min_y ... min_x/max_y)
    c = by_id["100"].corners
    assert c.shape == (4, 2)
    np.testing.assert_allclose(c, [[45, 666], [1874, 666], [1874, 999], [45, 999]])

    # split code mapping and filenames
    assert by_id["100"].split == "train" and by_id["101"].split == "test"
    assert by_id["100"].video_filename == "video_100.mp4"
    assert by_id["100"].label_filename == "audio_100.0.midi"

    # a custom video_pattern (server layout: video_<id>.0.mp4) is honoured
    recs2 = load_recordings_pianoyt(csv, video_pattern="video_{id}.0.mp4", valid_frac=0.0)
    assert {r.record_time: r.video_filename for r in recs2}["100"] == "video_100.0.mp4"

    # deterministic valid holdout carves from train only, and is reproducible
    r1 = load_recordings_pianoyt(csv, valid_frac=0.5, valid_seed=7)
    r2 = load_recordings_pianoyt(csv, valid_frac=0.5, valid_seed=7)
    valid1 = {r.record_time for r in r1 if r.split == "valid"}
    valid2 = {r.record_time for r in r2 if r.split == "valid"}
    assert valid1 == valid2 and len(valid1) >= 1
    assert valid1 <= {"100", "102"}  # only train ids can become valid, never test "101"


def test_frame_sampling_alignment():
    # 60 fps source -> 30 fps target = every other native frame (PianoVAM case:
    # must stay byte-identical to the old integer-stride behaviour).
    idx = plan_frame_sampling(600, 60.0, 30.0)
    assert len(idx) == 300
    np.testing.assert_array_equal(idx, np.arange(300) * 2)

    # 25 fps source -> 30 fps target: target frame k must land at real time
    # ~k/30 s (the old code left it at k/25 s -> drift). Check alignment holds
    # across the clip to within one native frame.
    idx = plan_frame_sampling(250, 25.0, 30.0)  # ~10 s of video
    for k in (0, 30, 90, len(idx) - 1):
        native_time = idx[k] / 25.0
        target_time = k / 30.0
        assert abs(native_time - target_time) <= 1.0 / 25.0 + 1e-9

    # max_frames caps the number of target frames.
    assert len(plan_frame_sampling(1000, 30.0, 30.0, max_frames=50)) == 50
    # never index past the end.
    idx = plan_frame_sampling(101, 29.97, 30.0)
    assert idx[-1] <= 100


# ------------------------------------------------------------ config / labels
def test_config_base_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "profiles:\n  dib:\n    hostname: nohost\n    overrides:\n      data.root: /a\n"
        "data:\n  root: /x\n  metadata: m.json\ntrain:\n  lr: 0.1\n  epochs: 3\n",
        encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        "base: base.yaml\nprofiles:\n  dib:\n    overrides:\n      data.strip_cache: /c\n"
        "train:\n  epochs: 9\n", encoding="utf-8")
    cfg = load_yaml(tmp_path / "child.yaml")
    assert cfg["train"] == {"lr": 0.1, "epochs": 9}          # merged, child wins
    assert cfg["data"]["metadata"] == "m.json"
    assert cfg["profiles"]["dib"]["overrides"] == {"data.root": "/a", "data.strip_cache": "/c"}
    cfg = load_config(tmp_path / "child.yaml", ["profile=dib", "train.lr=0.5"])
    assert cfg["data"]["root"] == "/a" and cfg["data"]["strip_cache"] == "/c"
    assert cfg["train"]["lr"] == 0.5


def test_shift_notes_and_label_offsets(tmp_path):
    notes = [Note(0.02, 0.5, 60, 80), Note(1.0, 1.4, 62, 80)]
    moved = shift_notes(notes, -0.1)                 # first note would start < 0
    assert [(round(n.onset, 3), round(n.offset, 3)) for n in moved] == [(0.9, 1.3)]
    assert shift_notes(notes, 0.0) is notes

    (tmp_path / "TSV").mkdir()
    (tmp_path / "TSV" / "r1.tsv").write_text(
        "#onset\tkey_offset\tframe_offset\tnote\tvelocity\n1.0\t1.5\t2.0\t60\t80\n",
        encoding="utf-8")
    offs = tmp_path / "offsets.json"
    offs.write_text(json.dumps({"r1": 0.25}), encoding="utf-8")
    rec = Recording("r1", "train", np.zeros((4, 2), np.float32))
    cfg = {"data": {"root": str(tmp_path), "tsv_dir": "TSV"},
           "labels": {"offset_field": "key_offset"}}
    assert read_reference(rec, cfg)[0].onset == 1.0
    cfg["data"]["label_offsets"] = str(offs)
    n = read_reference(rec, cfg)[0]
    assert (n.onset, n.offset) == (1.25, 1.75)


# ------------------------------------------------------------------ metrics
def test_paper_protocol_metrics_tolerances():
    pytest.importorskip("mir_eval")
    from pianovam_vision.metrics import note_scores_protocols

    ref = [Note(1.0 + i, 1.5 + i, 60 + i, 80) for i in range(10)]
    late = shift_notes(ref, 0.07)                     # every onset 70 ms late
    s = note_scores_protocols(ref, late)
    assert s["onset_f1@50"] == 0.0 and s["onset_f1@100"] == 1.0
    assert s["offtol_f1@100"] == 1.0 and s["offtol_f1@50"] == 0.0
    perfect = note_scores_protocols(ref, ref)
    assert all(abs(perfect[k] - 1.0) < 1e-9 for k in perfect)


def test_sync_lag_sign_convention():
    """Labels 100 ms LATE vs the video -> best lag -0.1 s, and applying it via
    shift_notes (what data.label_offsets does) lines the labels up again."""
    pytest.importorskip("mir_eval")
    pytest.importorskip("torch")
    from pianovam_vision.sync_check import estimate_lag

    video = [Note(0.5 + 0.37 * i, 0.7 + 0.37 * i, 40 + i % 30, 80) for i in range(60)]
    # model onsets are quantised to the 30 fps frame grid, like real predictions
    est = [Note(round(n.onset * 30) / 30, round(n.offset * 30) / 30, n.pitch, 80)
           for n in video]
    labels = shift_notes(video, 0.10)
    lags = np.arange(-0.4, 0.4001, 0.01)
    best, _ = estimate_lag(labels, est, lags, 0.05)
    assert abs(best - (-0.10)) < 0.01          # centre of the plateau, not its edge
    fixed = shift_notes(labels, best)
    assert max(abs(a.onset - b.onset) for a, b in zip(fixed, video)) < 0.02


# ------------------------------------------------------------- lr schedule
def test_lr_schedule_warmup_cosine():
    pytest.importorskip("torch")
    from pianovam_vision.train import lr_at

    t = {"lr": 1.0}
    assert lr_at(0, 100, t) == 1.0 and lr_at(99, 100, t) == 1.0     # constant default
    t = {"lr": 1.0, "lr_schedule": "cosine", "warmup_frac": 0.1, "min_lr_ratio": 0.0}
    assert lr_at(0, 100, t) == pytest.approx(0.1)                   # warming up
    assert lr_at(9, 100, t) == pytest.approx(1.0)                   # peak
    assert lr_at(55, 100, t) == pytest.approx(0.5, abs=0.02)        # half way down
    assert lr_at(100, 100, t) == pytest.approx(0.0, abs=1e-9)       # fully decayed


# -------------------------------------------------------- strip cache + aug
def _cache_cfg(tmp_path, corners, grayscale=False):
    return {"data": {"root": str(tmp_path), "video_dir": "Video", "video_ext": ".mp4",
                     "strip_cache": str(tmp_path / "cache")},
            "keyboard": {"warp_width": 64, "warp_height": 16, "grayscale": grayscale,
                         "decode_height": 360},
            "labels": {"fps": 30.0},
            "train": {"max_frames_per_record": 0}}


def test_strip_cache_roundtrip_and_staleness(tmp_path):
    pytest.importorskip("cv2")
    from pianovam_vision.strip_cache import (
        expected_meta, open_reader, try_open_cached, write_record_cache)

    # Smooth frames (like real keyboard strips) with a pure-red patch whose
    # position encodes the frame index -> checks order, RGB channel order, content.
    grad = np.linspace(40, 200, 64, dtype=np.float32)[None, :, None]
    strips = []
    for i in range(20):
        s = np.repeat(np.repeat(grad, 16, axis=0), 3, axis=2).astype(np.uint8)
        s[4:12, 2 * i + 4:2 * i + 12] = (255, 0, 0)
        strips.append(s)
    corners = np.array([[0, 0], [100, 0], [100, 20], [0, 20]], np.float32)
    rec = Recording("r1", "train", corners)
    cfg = _cache_cfg(tmp_path, corners)
    write_record_cache(strips, cfg["data"]["strip_cache"], "r1",
                       expected_meta(cfg, rec), quality=95)

    reader, why = try_open_cached(cfg, rec)
    assert why is None and len(reader) == 20 and reader.source == "cache"
    got = reader.read_warped([3, 4, 5, 9, 9, 19, 25])           # runs, repeats, clip
    assert got.shape == (7, 16, 64, 3) and got.dtype == np.uint8
    for g, i in zip(got, [3, 4, 5, 9, 9, 19, 19]):
        assert np.abs(g.astype(int) - strips[i].astype(int)).mean() < 6   # JPEG-close
        r, gg, b = g[8, 2 * i + 8].astype(int)                           # patch centre
        assert r > 200 and gg < 60 and b < 60                            # still red (RGB)

    assert len(open_reader(cfg, rec, max_frames=8)) == 8          # cap honoured
    gray = try_open_cached(_cache_cfg(tmp_path, corners, grayscale=True), rec)[0]
    assert gray.read_warped([0]).shape == (1, 16, 64, 1)          # colour cache -> gray

    moved = Recording("r1", "train", corners + 5)                 # crop box changed
    assert try_open_cached(cfg, moved)[0] is None
    cfg2 = _cache_cfg(tmp_path, corners)
    cfg2["keyboard"]["decode_height"] = 720                       # decode setting changed
    assert "decode_height" in try_open_cached(cfg2, rec)[1]
    assert try_open_cached(cfg, Recording("r2", "train", corners))[1] == "not cached"


def test_augment_identity_and_ranges():
    pytest.importorskip("cv2")
    torch = pytest.importorskip("torch")
    from pianovam_vision import augment as aug

    rng = np.random.default_rng(0)
    m = aug.sample_affine(1408, 112, {}, rng)                      # all-zero config
    np.testing.assert_allclose(m, [[1, 0, 0], [0, 1, 0]], atol=1e-6)
    frames = rng.integers(0, 255, (4, 112, 1408, 3), dtype=np.uint8)
    np.testing.assert_array_equal(aug.apply_geometric(frames, m), frames)

    # With the shipped config values, no key (not even A0/C8 at the strip ends)
    # ever moves more than half a 16-px key column, so labels stay correct.
    from pathlib import Path
    for name in ("vam_full.yaml", "yt_full.yaml"):
        a = load_yaml(Path(__file__).parent.parent / "configs" / name)["augment"]
        for _ in range(200):
            m = aug.sample_affine(1408, 112, a, rng)
            for x in (0.0, 704.0, 1408.0):
                for y in (0.0, 112.0):
                    dx = m[0, 0] * x + m[0, 1] * y + m[0, 2] - x
                    assert abs(dx) < 8.0, (name, x, y, dx)

    x = torch.rand(4, 3, 8, 32)
    y = aug.apply_photometric(x.clone(), {"brightness": 0.5, "contrast": 0.5,
                                          "gray_p": 1.0, "noise_std": 0.1, "noise_p": 1.0}, rng)
    assert y.shape == x.shape and float(y.min()) >= 0.0 and float(y.max()) <= 1.0
