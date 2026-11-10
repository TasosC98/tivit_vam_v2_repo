"""Tests for labels, decoding, configs, caching, metrics and models (no GPU or video needed).

    python -m pytest tests -q

Tests needing torch / OpenCV / mir_eval are skipped where those are missing.
"""
from pathlib import Path

import numpy as np
import pytest

from pianovam_vision.config import load_config, load_yaml
from pianovam_vision.labels import Note, build_target_rolls
from pianovam_vision.decode import decode_notes
from pianovam_vision.metadata import Recording, load_recordings_pianoyt
from pianovam_vision.video import plan_frame_sampling

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


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


def test_decode_velocity_is_read_at_the_onset_frame():
    on, fr, vel = np.zeros((40, 88)), np.zeros((40, 88)), np.zeros((40, 88))
    on[10:12, 39] = 0.95
    fr[10:20, 39] = 0.9
    vel[10, 39], vel[11, 39] = 0.5, 0.9
    (n,) = decode_notes(on, fr, 30.0, velocity_probs=vel)
    assert (n.pitch, n.velocity) == (60, round(0.5 * 127))
    (n,) = decode_notes(on, fr, 30.0)
    assert n.velocity == 80                                   # default without a velocity head


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

    # a custom video_pattern (layout video_<id>.0.mp4) is honoured
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
    # 60 fps source -> 30 fps target = every other native frame (PianoVAM case).
    idx = plan_frame_sampling(600, 60.0, 30.0)
    assert len(idx) == 300
    np.testing.assert_array_equal(idx, np.arange(300) * 2)

    # 25 fps source -> 30 fps target: target frame k must land at real time
    # ~k/30 s. Check alignment holds across the clip to within one native frame.
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


# ------------------------------------------------------------------ configs
def test_config_base_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "data:\n  root: /x\n  metadata: m.json\ntrain:\n  lr: 0.1\n  epochs: 3\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("base: base.yaml\ntrain:\n  epochs: 9\n", encoding="utf-8")
    cfg = load_yaml(tmp_path / "child.yaml")
    assert cfg["train"] == {"lr": 0.1, "epochs": 9}          # merged, child wins
    assert cfg["data"]["metadata"] == "m.json"
    cfg = load_config(tmp_path / "child.yaml", ["train.lr=0.5", "data.root=/y", "model.use_velocity=true"])
    assert cfg["train"]["lr"] == 0.5 and cfg["data"]["root"] == "/y"
    assert cfg["model"]["use_velocity"] is True


def test_paper_configs():
    """The variants differ from the KeyWise config only where the paper says they do."""
    base = load_config(CONFIGS / "pianovam.yaml")
    assert base["model"]["arch"] == "tiled" and base["keyboard"]["warp_width"] == 88 * 16
    vel = load_config(CONFIGS / "pianovam_velocity.yaml")
    assert vel["model"]["use_velocity"] is True and vel["train"]["velocity_loss_weight"] == 1.0
    for name, bins in (("pianovam_global.yaml", 16), ("pianovam_global88.yaml", 88)):
        g = load_config(CONFIGS / name)
        assert g["model"]["arch"] == "strip" and g["model"]["strip_bins"] == bins
        assert {k: v for k, v in g.items() if k not in ("model", "train")} == \
               {k: v for k, v in base.items() if k not in ("model", "train")}
    yt = load_config(CONFIGS / "pianoyt.yaml")
    for block in ("keyboard", "labels", "model", "decode", "infer", "augment"):
        assert yt[block] == base[block], block
    assert {k: v for k, v in yt["train"].items() if k != "out_dir"} == \
           {k: v for k, v in base["train"].items() if k != "out_dir"}
    ft = load_config(CONFIGS / "pianoyt_finetune.yaml")
    assert ft["train"]["init_from"] and ft["train"]["lr"] == 0.0002 and ft["train"]["epochs"] == 12


# ------------------------------------------------------------------ metrics
def test_paper_protocol_metrics_tolerances():
    pytest.importorskip("mir_eval")
    from pianovam_vision.metrics import note_scores_protocols

    ref = [Note(1.0 + i, 1.5 + i, 60 + i, 80) for i in range(10)]
    late = [Note(n.onset + 0.07, n.offset + 0.07, n.pitch, n.velocity) for n in ref]   # 70 ms late
    s = note_scores_protocols(ref, late)
    assert s["onset_f1@50"] == 0.0 and s["onset_f1@100"] == 1.0
    assert s["offtol_f1@100"] == 1.0 and s["offtol_f1@50"] == 0.0
    perfect = note_scores_protocols(ref, ref)
    assert all(abs(perfect[k] - 1.0) < 1e-9 for k in perfect)


def test_note_scores_identical_to_mir_eval():
    """Scoring pitch by pitch must give exactly mir_eval's numbers."""
    mir_eval = pytest.importorskip("mir_eval")
    from pianovam_vision.metrics import _to_intervals, note_prf

    rng = np.random.default_rng(0)

    def notes(n):
        on = np.sort(rng.random(n) * 30)
        dur = rng.random(n) + 0.01
        return [Note(float(o), float(o + d), int(p), 80)
                for o, d, p in zip(on, dur, rng.integers(55, 62, n))]

    for _ in range(20):
        ref, est = notes(int(rng.integers(1, 200))), notes(int(rng.integers(1, 200)))
        ri, rf = _to_intervals(ref)
        ei, ef = _to_intervals(est)
        for tol in (0.05, 0.10):
            for kw in (dict(offset_ratio=None), dict(offset_ratio=0.2, offset_min_tolerance=0.05),
                       dict(offset_ratio=0.0, offset_min_tolerance=tol)):
                want = mir_eval.transcription.precision_recall_f1_overlap(
                    ri, rf, ei, ef, onset_tolerance=tol, **kw)[:3]
                assert note_prf(ref, est, onset_tolerance=tol, **kw) == want


def test_velocity_scores_identical_to_mir_eval():
    """V2N's ++Vel / ++Off++Vel: per-pitch matching + one global velocity fit = mir_eval."""
    mir_eval = pytest.importorskip("mir_eval")
    from pianovam_vision.metrics import _to_intervals, note_velocity_prf

    rng = np.random.default_rng(3)
    for _ in range(30):
        ref = [Note(float(o), float(o + d), int(p), int(v)) for o, d, p, v in zip(
            np.sort(rng.random(150) * 30), rng.random(150) + 0.01, rng.integers(55, 62, 150),
            rng.integers(20, 120, 150))]
        est = [Note(n.onset + 0.02 * rng.random(), n.offset + 0.03, n.pitch,
                    int(np.clip(0.7 * n.velocity + 20 + rng.normal(0, 8), 1, 127)))
               for n in ref if rng.random() < 0.8]
        ri, rf = _to_intervals(ref)
        ei, ef = _to_intervals(est)
        rv = np.array([n.velocity for n in ref], float)
        ev = np.array([n.velocity for n in est], float)
        for tol in (0.05, 0.10):
            for kw in (dict(offset_ratio=None), dict(offset_ratio=0.0, offset_min_tolerance=tol)):
                for scale in (None, 127.0):       # MIDI velocities / V2N's velocity / 127
                    s = scale or 1.0
                    want = mir_eval.transcription_velocity.precision_recall_f1_overlap(
                        ri, rf, rv / s, ei, ef, ev / s, onset_tolerance=tol, **kw)[:3]
                    got = note_velocity_prf(ref, est, onset_tolerance=tol, velocity_scale=scale, **kw)
                    assert np.allclose(got, want, atol=1e-12)


def test_calibration_count_and_bound_are_exact():
    """The fast threshold search relies on two facts: (1) the number of decoded
    notes equals the number of rising edges of the thresholded onset roll, and
    (2) F1 <= 2*min(n_ref, n_est) / (n_ref + n_est). Both must hold exactly."""
    pytest.importorskip("mir_eval")
    from pianovam_vision.metrics import note_scores

    rng = np.random.default_rng(0)
    fps = 30.0
    for trial in range(20):
        T = 300
        onset_p = rng.random((T, 88)) ** 8          # mostly low, some spikes
        frame_p = rng.random((T, 88))
        ot = float(rng.choice([0.05, 0.2, 0.5, 0.8]))
        b = onset_p >= ot
        count = int(b[0].sum()) + int((b[1:] & ~b[:-1]).sum())
        est = decode_notes(onset_p, frame_p, fps, onset_threshold=ot,
                           frame_threshold=0.5, min_duration_s=0.03)
        assert count == len(est)

        ref = [Note(float(t) / fps, float(t) / fps + 0.3, int(p), 80)
               for t, p in zip(rng.integers(0, T - 10, 60), rng.integers(21, 109, 60))]
        ref.sort(key=lambda n: (n.onset, n.pitch))
        s = note_scores(ref, est)
        bound = 2 * min(len(ref), len(est)) / (len(ref) + len(est))
        assert s["onset_f1"] <= bound + 1e-9 and s["full_f1"] <= bound + 1e-9


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

    # A mixed / truncated set (interrupted or overlapping builds) is rejected ...
    from pianovam_vision.strip_cache import cache_paths, remove_partial_files
    bin_p = cache_paths(cfg["data"]["strip_cache"], "r1")[0]
    data = bin_p.read_bytes()
    bin_p.write_bytes(data[:-10])
    assert "incomplete" in try_open_cached(cfg, rec)[1]
    bin_p.write_bytes(data)
    assert try_open_cached(cfg, rec)[1] is None
    # ... and leftover temporary files are cleaned up.
    for name in ("r1.bin.tmp", "r1.idx.npy.tmp4242"):
        (bin_p.parent / name).write_bytes(b"x" * 1000)
    assert remove_partial_files(cfg["data"]["strip_cache"]) > 0
    assert not list(bin_p.parent.glob("*.tmp*"))
    assert try_open_cached(cfg, rec)[1] is None           # real files untouched


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
    for name in ("pianovam.yaml", "pianoyt.yaml"):
        a = load_config(CONFIGS / name)["augment"]
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


# ------------------------------------------------------------------ models
def test_models_parameters_and_mac_count():
    """The parameter counts and MACs of the paper, for every model config."""
    torch = pytest.importorskip("torch")
    from pianovam_vision.model import build_model
    from pianovam_vision.speed import count_macs

    keywise = build_model(load_config(CONFIGS / "pianovam.yaml"))
    assert sum(p.numel() for p in keywise.parameters()) == 1_299_491
    x = torch.rand(1, 2, 3, 112, 1408)
    assert abs(sum(count_macs(keywise, x).values()) - 1.868e9) < 0.001e9

    vel = build_model(load_config(CONFIGS / "pianovam_velocity.yaml"))
    assert sum(p.numel() for p in vel.parameters()) == 1_299_491 + 257
    with torch.no_grad():
        v = vel(torch.rand(1, 3, 3, 112, 1408))["velocity"]
    assert v.shape == (1, 3, 88) and float(v.min()) >= 0.0 and float(v.max()) <= 1.0

    glob = build_model(load_config(CONFIGS / "pianovam_global.yaml"))
    assert sum(p.numel() for p in glob.parameters()) == 3_698_128
    assert glob.encoder.fc[1].in_features == 192 * 16

    g88 = build_model(load_config(CONFIGS / "pianovam_global88.yaml"))
    assert sum(p.numel() for p in g88.parameters()) == 7_237_072
    assert g88.encoder.fc[1].in_features == 192 * 88
    out = g88(torch.rand(1, 3, 3, 112, 1408))
    assert out["onset_logits"].shape == (1, 3, 88) and out["frame_logits"].shape == (1, 3, 88)


def test_lstm_mac_count():
    """The MAC counter counts an LSTM (V2N's heads) as 4 gates x hidden x (input + hidden)
    per step and direction."""
    torch = pytest.importorskip("torch")
    from pianovam_vision.speed import count_macs

    class Seq(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.LSTM(10, 20, batch_first=True, bidirectional=True)

        def forward(self, x):
            return self.rnn(x)[0]

    assert count_macs(Seq(), torch.rand(2, 7, 10))["rnn"] == 2 * 4 * 20 * (10 + 20)
