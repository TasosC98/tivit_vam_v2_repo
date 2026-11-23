"""R3 support: keyboard geometry, folder loader, valid split, MIDI labels, configs,
per-subset reporting. No video or GPU needed.

    cd vision_transcription && python -m pytest tests/test_r3.py -q
"""
import json
from pathlib import Path

import numpy as np
import pytest

from pianovam_vision.config import load_config, load_yaml
from pianovam_vision.labels import read_reference, reference_path
from pianovam_vision.metadata import load_recordings_r3, r3_box_corners, r3_date

ROOT = Path(__file__).resolve().parents[1]

# A real R3 box (r3s_bounding_boxes.json, session rach3_2020-10-09_a01_full).
BOX = {"rot_angle": "3.0", "x1": "317", "y1": "499", "x2": "1556", "y2": "643"}


def test_box_corners_rotation_sign_and_180_degree_view():
    # Reference: OpenCV, invertAffineTransform(getRotationMatrix2D((960, 540), -3, 1))
    # applied to the box; this sign is the one that fits the keyboard in the frames.
    want_img = np.array([[315.74, 532.71], [1553.04, 467.86],      # image TL, TR
                         [1560.57, 611.67], [323.27, 676.51]])     # image BR, BL
    got = r3_box_corners(BOX)
    # rot180: keyboard LT (A0, key backs) = image BR, RT = BL, RB = TL, LB = TR
    assert np.allclose(got, want_img[[2, 3, 0, 1]], atol=0.01)
    assert got[0, 0] > got[1, 0]                   # A0 end is on the image's right
    assert np.allclose(r3_box_corners(BOX, orientation="normal"), want_img, atol=0.01)


def test_box_corners_without_rotation_are_the_box():
    box = {"rot_angle": "0.0", "x1": "558", "y1": "642", "x2": "1436", "y2": "738"}
    c = r3_box_corners(box)
    assert np.allclose(c, [[1436, 738], [558, 738], [558, 642], [1436, 642]])
    with pytest.raises(ValueError):
        r3_box_corners(box, orientation="sideways")


def test_r3_date():
    assert r3_date("ls_2024-02-06_a02") == "2024-02-06"
    assert r3_date("rach3_2020-10-09_a01") == "2020-10-09"


def _box(rot="0.0"):
    return {"rot_angle": rot, "x1": "500", "y1": "600", "x2": "1400", "y2": "700"}


def make_tree(root: Path, chunks: dict) -> Path:
    """chunks: {folder: [(session, n_chunks)]} -> empty mp4/mid files + box files."""
    boxes = {"r3s": {}, "r3x": {}}
    for folder, sessions in chunks.items():
        sub = "r3s" if folder.startswith("r3s") else "r3x"
        (root / folder).mkdir(parents=True, exist_ok=True)
        for session, n in sessions:
            boxes[sub][f"{session}_full"] = _box()
            for k in range(1, n + 1):
                (root / folder / f"{session}_full_split{k}.mp4").write_bytes(b"")
                (root / folder / f"{session}_split{k}.mid").write_bytes(b"")
    for sub, b in boxes.items():
        (root / f"{sub}_bounding_boxes.json").write_text(json.dumps(b))
    return root


TREE = {
    "r3s_train": [("rach3_2021-01-0%d_a01" % d, 2) for d in range(1, 10)],
    "r3s_test": [("rach3_2022-05-05_a01", 3)],
    "train": [("warmup_2023-01-%02d_a01" % d, 3) for d in range(10, 30)]
             + [("ls_2024-02-12_a02", 4), ("ls_2024-02-12_a03", 1)],
    "test": [("ls_2024-02-06_a02", 2)],
}


def test_loader_paths_splits_and_valid_by_date(tmp_path):
    root = make_tree(tmp_path, TREE)
    recs = load_recordings_r3(root, valid_frac=0.2, valid_seed=7,
                              valid_avoid_sessions=["ls_2024-02-12_a02"])
    by = {r.record_time: r for r in recs}
    assert len(recs) == 9 * 2 + 3 + 20 * 3 + 5 + 2
    r = by["ls_2024-02-06_a02_split2"]
    assert (r.subset, r.split, r.session) == ("r3x", "test", "ls_2024-02-06_a02")
    assert r.video_path(root, "", ".mp4") == root / "test" / "ls_2024-02-06_a02_full_split2.mp4"
    assert r.label_path(root, "") == root / "test" / "ls_2024-02-06_a02_split2.mid"
    assert r.corners.shape == (4, 2) and r.corners.dtype == np.float32
    for sub, n_train in (("r3s", 18), ("r3x", 65)):
        valid = [x for x in recs if x.subset == sub and x.split == "valid"]
        assert 0 < len(valid) <= 2 * 0.2 * n_train
        # whole dates only: no date is split between train and valid
        vd = {r3_date(x.session) for x in valid}
        assert not any(r3_date(x.session) in vd for x in recs
                       if x.subset == sub and x.split == "train")
    # test chunks never move; avoided sessions (and their whole date) stay in train
    assert all(x.split == "test" for x in recs if "test" in x.label_filename.split("/")[0])
    assert all(x.split == "train" for x in recs if x.session.startswith("ls_2024-02-12"))
    # deterministic, and one subset's choice does not depend on the other
    again = load_recordings_r3(root, valid_frac=0.2, valid_seed=7,
                               valid_avoid_sessions=["ls_2024-02-12_a02"])
    assert [x.split for x in again] == [x.split for x in recs]
    only_x = load_recordings_r3(root, subsets=["r3x"], valid_frac=0.2, valid_seed=7,
                                valid_avoid_sessions=["ls_2024-02-12_a02"])
    assert {x.subset for x in only_x} == {"r3x"}
    assert {x.record_time: x.split for x in only_x} == \
           {x.record_time: x.split for x in recs if x.subset == "r3x"}


def test_loader_errors_name_the_problem(tmp_path):
    root = make_tree(tmp_path, {"r3s_train": [("rach3_2021-01-01_a01", 1)]})
    with pytest.raises(FileNotFoundError, match="r3s_test"):
        load_recordings_r3(root)
    (root / "r3s_test").mkdir()
    (root / "r3s_test" / "rach3_2022-01-01_a01_split1.mid").write_bytes(b"")
    with pytest.raises(KeyError, match="rach3_2022-01-01_a01_full"):
        load_recordings_r3(root, subsets=["r3s"])


def test_r3_config_is_the_paper_recipe():
    r3, vam = load_yaml(ROOT / "configs" / "r3.yaml"), load_yaml(ROOT / "configs" / "vam_full.yaml")
    for block in ("model", "labels", "decode", "infer", "augment"):
        assert r3[block] == vam[block], block
    assert r3["keyboard"] == dict(vam["keyboard"], decode_height=1080)
    assert r3["train"] == dict(vam["train"], out_dir="runs/r3_full")
    assert r3["data"]["format"] == "r3" and r3["data"]["exclude_records"] == []
    assert r3["data"]["cache_reader"] == "cv2" and "cache_reader" not in vam["data"]
    # r3c.yaml (hand-set corners) = r3.yaml + the corners file + its own cache folder
    r3c = load_yaml(ROOT / "configs" / "r3c.yaml")
    for block in ("model", "labels", "decode", "infer", "augment", "keyboard"):
        assert r3c[block] == r3[block], block
    assert r3c["train"] == dict(r3["train"], out_dir="runs/r3c_full")
    assert r3c["data"] == dict(r3["data"], r3_corners="configs/r3_corners.json")
    cache = lambda c: c["profiles"]["dib"]["overrides"]["data.strip_cache"]
    assert cache(r3c) != cache(r3) and r3c["profiles"]["dib"]["overrides"]["data.root"] == \
        r3["profiles"]["dib"]["overrides"]["data.root"]
    for name, change in (("r3c_vel", {"use_velocity": True}),
                         ("r3c_global88", {"arch": "strip", "strip_bins": 88})):
        c = load_yaml(ROOT / "configs" / f"{name}.yaml")
        assert c["model"] == dict(r3["model"], **change) and c["data"] == r3c["data"]
    ft = load_yaml(ROOT / "configs" / "r3c_finetune.yaml")
    assert ft["data"] == r3c["data"]
    assert ft["train"]["init_from"].endswith("best.pt") and ft["train"]["lr"] < r3["train"]["lr"]
    ns = load_yaml(ROOT / "configs" / "r3_nosync.yaml")["data"]["exclude_records"]
    assert len(ns) == 10 and all(x.startswith("ls_2024-02-06_a0") for x in ns)


@pytest.mark.parametrize("name", ["r3_corners_hand.json", "r3_corners.json"])
def test_r3_corners_file_is_complete_and_oriented(name):
    """144 sessions, each a keyboard seen turned by 180 deg (hand-set and refined)."""
    from pianovam_vision.metadata import load_r3_corners

    c = load_r3_corners(ROOT / "configs" / name)
    assert len(c) == 144
    assert sum(s.startswith("rach3_") for s in c) == 27       # R3s sessions
    for s, (a0b, c8b, c8f, a0f) in c.items():
        q = np.array([a0b, c8b, c8f, a0f])
        assert ((q >= 0) & (q <= [1920, 1080])).all(), s
        assert a0b[0] > c8b[0] and a0f[0] > c8f[0], s          # rot180: A0 at the right
        assert a0b[1] > a0f[1] and c8b[1] > c8f[1], s          # key backs below the fronts
        assert 700 < a0b[0] - c8b[0] < 1350, s                 # 40-65 % of the width


def test_refined_corners_stay_close_to_the_hand_set_ones():
    from pianovam_vision.metadata import load_r3_corners

    hand = load_r3_corners(ROOT / "configs" / "r3_corners_hand.json")
    ref = load_r3_corners(ROOT / "configs" / "r3_corners.json")
    kept = json.loads((ROOT / "configs" / "r3_corners.json").read_text())["refined"]["kept_as_set_by_hand"]
    assert kept == ["rach3_2021-08-19_a01"] and set(hand) == set(ref)
    moves = {s: np.abs(ref[s] - hand[s]).max() for s in hand}
    assert all(moves[s] < 0.01 for s in kept)
    assert max(moves.values()) <= 10.0 + 0.01                 # refine's bound
    assert np.median(list(moves.values())) < 2.0


def write_midi(path: Path, notes, pedal=True):
    pretty_midi = pytest.importorskip("pretty_midi")
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for on, off, p in notes:
        inst.notes.append(pretty_midi.Note(velocity=70, pitch=p, start=on, end=off))
    if pedal:   # sustain pedal held over everything: must NOT lengthen the notes
        inst.control_changes += [pretty_midi.ControlChange(64, 127, 0.0),
                                 pretty_midi.ControlChange(64, 0, 9.0)]
    pm.instruments.append(inst)
    pm.write(str(path))


def r3_cfg(root: Path):
    cfg = load_config(ROOT / "configs" / "r3.yaml", [f"data.root={root.as_posix()}"])
    cfg["data"]["root"] = str(root)        # no server profile on a test machine
    return cfg


def test_r3_labels_are_key_releases(tmp_path):
    root = make_tree(tmp_path, {k: v[:1] for k, v in TREE.items()})
    cfg = r3_cfg(root)
    from pianovam_vision.metadata import recordings_from_cfg
    rec = next(r for r in recordings_from_cfg(cfg) if r.split == "test" and r.subset == "r3x")
    path = reference_path(rec, cfg)
    assert path == root / "test" / f"{rec.record_time}.mid"
    write_midi(path, [(1.0, 1.2, 60), (1.5, 1.52, 64)])
    notes = read_reference(rec, cfg)
    assert [(round(n.onset, 3), round(n.offset, 3), n.pitch) for n in notes] == \
           [(1.0, 1.2, 60), (1.5, 1.52, 64)]


def test_score_midi_reports_each_subset(tmp_path, capsys, monkeypatch):
    pytest.importorskip("mir_eval")
    pytest.importorskip("scipy")
    root = make_tree(tmp_path / "r3", {k: v[:2] for k, v in TREE.items()})
    cfg = r3_cfg(root)
    from pianovam_vision.metadata import recordings_from_cfg
    from pianovam_vision import score_midi
    pred, base = tmp_path / "pred", tmp_path / "base"
    pred.mkdir(), base.mkdir()
    tests = [r for r in recordings_from_cfg(cfg) if r.split == "test"]
    assert {r.subset for r in tests} == {"r3s", "r3x"}
    for i, r in enumerate(tests):
        notes = [(0.5 + 0.4 * k, 0.8 + 0.4 * k, 50 + (k + i) % 30) for k in range(12)]
        write_midi(reference_path(r, cfg), notes)
        write_midi(pred / f"{r.record_time}.mid", notes, pedal=False)            # perfect
        write_midi(base / f"{r.record_time}.mid", notes[: 6 + i % 5], pedal=False)  # misses
    monkeypatch.setattr("sys.argv", ["score_midi", "--config", str(ROOT / "configs" / "r3.yaml"),
                                     "--split", "test", "--pred", str(pred),
                                     "--baseline", str(base), f"data.root={root.as_posix()}"])
    score_midi.main()
    out = capsys.readouterr().out
    assert f"mean over {len(tests)} recordings" in out
    n_s = sum(r.subset == "r3s" for r in tests)
    assert f"subset r3s: mean over {n_s} recordings" in out
    assert f"subset r3x: mean over {len(tests) - n_s} recordings" in out
    line = next(x for x in out.splitlines() if x.startswith("Onset F1 @50 ms"))
    assert line.split()[4] == "100.0"                      # pred column


def test_evaluate_prints_subset_means(capsys):
    pytest.importorskip("torch")
    from pianovam_vision.evaluate import print_subsets
    keys = ["onset_f1@50", "onset_f1@100", "offtol_f1@50", "offtol_f1@100", "frame_f1"]
    rows = [dict({k: 0.5 for k in keys}, subset="r3s"), dict({k: 0.9 for k in keys}, subset="r3x"),
            dict({k: 0.7 for k in keys}, subset="r3x")]
    print_subsets(rows)
    out = capsys.readouterr().out
    assert "r3s" in out and "50.00" in out and "80.00" in out
    print_subsets([dict({k: 0.5 for k in keys})])          # PianoVAM rows: nothing printed
    assert capsys.readouterr().out == ""


# --- strip cache built with OpenCV (data.cache_reader: cv2) -------------------------

def write_video(path: Path, n: int, fps: float, size=(320, 180)):
    """Frame i: grey level 8*i, the left quarter pure red (checks RGB order)."""
    cv2 = pytest.importorskip("cv2")
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not w.isOpened():
        pytest.skip("OpenCV cannot write mp4 here")
    for i in range(n):
        f = np.full((size[1], size[0], 3), 8 * i, np.uint8)
        f[:, : size[0] // 4] = (0, 0, 255)                     # BGR red
        w.write(f)
    w.release()


def test_sequential_reader_follows_the_frame_plan(tmp_path):
    from pianovam_vision.video import SequentialWarpedVideo
    p = tmp_path / "v.mp4"
    write_video(p, 20, 60.0)
    whole = np.float32([[0, 0], [320, 0], [320, 180], [0, 180]])
    v = SequentialWarpedVideo(p, whole, 64, 16, 30.0)
    assert (v.native_len, v.native_fps, len(v)) == (20, 60.0, 10)   # 60 -> 30 fps
    strips = list(v.strips())
    assert len(strips) == 10 and strips[0].shape == (16, 64, 3) and strips[0].dtype == np.uint8
    grey = [float(s[:, 40:, 1].mean()) for s in strips]
    assert np.allclose(grey, [16 * k for k in range(10)], atol=4)   # native frames 0, 2, 4, ...
    red = strips[5][:, :12].reshape(-1, 3).mean(axis=0)
    assert red[0] > 200 and red[2] < 60                              # RGB out, like decord
    small = list(SequentialWarpedVideo(p, whole, 64, 16, 30.0, decode_height=90).strips())
    assert np.abs(np.float32(small) - np.float32(strips)).mean() < 3  # resize keeps the content


def _real_tree(tmp_path: Path) -> Path:
    """Four folders, one chunk each; the two test chunks get real 320x180 videos."""
    root = make_tree(tmp_path / "r3", {"r3s_train": [("rach3_2021-01-01_a01", 1)],
                                       "r3s_test": [("rach3_2022-05-05_a01", 1)],
                                       "train": [("warmup_2023-01-10_a01", 1)],
                                       "test": [("ls_2024-02-06_a02", 1)]})
    box = {"rot_angle": "0.0", "x1": "40", "y1": "60", "x2": "280", "y2": "120"}
    for sub in ("r3s", "r3x"):
        f = root / f"{sub}_bounding_boxes.json"
        f.write_text(json.dumps({k: box for k in json.loads(f.read_text())}))
    write_video(root / "r3s_test" / "rach3_2022-05-05_a01_full_split1.mp4", 12, 30.0)
    write_video(root / "test" / "ls_2024-02-06_a02_full_split1.mp4", 9, 30.0)
    return root


def test_build_cache_with_opencv_reader(tmp_path, capsys, monkeypatch):
    pytest.importorskip("cv2")
    from pianovam_vision import build_cache
    from pianovam_vision.metadata import recordings_from_cfg
    from pianovam_vision.strip_cache import cache_paths, try_open_cached
    from pianovam_vision.video import SequentialWarpedVideo
    root, cache = _real_tree(tmp_path), tmp_path / "cache"
    ov = [f"data.root={root.as_posix()}", f"data.strip_cache={cache.as_posix()}",
          "data.frame_size=[320, 180]"]
    argv = ["build_cache", "--config", str(ROOT / "configs" / "r3.yaml"), "--splits", "test",
            "--workers", "1", "--min_free_gb", "0", *ov]
    monkeypatch.setattr("sys.argv", argv)
    build_cache.main()
    assert "0 already cached / skipped, 2 to build" in capsys.readouterr().out

    cfg = load_config(ROOT / "configs" / "r3.yaml", ov)
    cfg["data"]["root"] = str(root)
    rec = next(r for r in recordings_from_cfg(cfg) if r.record_time == "rach3_2022-05-05_a01_split1")
    cached, why = try_open_cached(cfg, rec)
    assert cached is not None, why
    assert cached.meta["reader"] == "cv2" and len(cached) == 12
    kb = cfg["keyboard"]
    direct = np.stack(list(SequentialWarpedVideo(
        root / "r3s_test" / "rach3_2022-05-05_a01_full_split1.mp4", rec.corners,
        kb["warp_width"], kb["warp_height"], 30.0, kb["decode_height"]).strips()))
    got = cached.read_warped(np.arange(12))
    assert got.shape == (12, 112, 1408, 3)
    assert np.abs(got.astype(np.int16) - direct).mean() < 2         # JPEG noise only

    # a cache written by decord (the first R3 attempt) is rebuilt; a cv2 one is kept
    meta_p = cache_paths(cache, rec.record_time)[2]
    meta = json.loads(meta_p.read_text())
    meta_p.write_text(json.dumps(dict(meta, reader="decord")))
    build_cache.main()
    assert "1 already cached / skipped, 1 to build" in capsys.readouterr().out
    assert json.loads(meta_p.read_text())["reader"] == "cv2"

    # --compare: OpenCV vs the cache, shift 0 = same frames
    monkeypatch.setattr("sys.argv", ["build_cache", "--config", str(ROOT / "configs" / "r3.yaml"),
                                     "--compare", rec.record_time, *ov])
    build_cache.main()
    out = capsys.readouterr().out
    zero = next(x for x in out.splitlines() if "shift +0" in x)
    assert float(zero.split("mean abs diff")[1].split(",")[0]) < 2


# --- annotated keyboard corners (data.r3_corners; corners.py export) ---------------

def test_loader_uses_annotated_corners_for_every_session(tmp_path):
    from pianovam_vision.metadata import load_r3_corners
    root = make_tree(tmp_path / "r3", {k: v[:1] for k, v in TREE.items()})
    official = load_recordings_r3(root, valid_frac=0)
    sessions = sorted({r.session for r in official})
    quad = [[1400.0, 700.0], [500.0, 701.5], [499.0, 601.0], [1401.0, 600.0]]
    f = tmp_path / "corners.json"
    f.write_text(json.dumps({"format": "r3_corners_v1", "corners": {s: quad for s in sessions}}))
    recs = load_recordings_r3(root, valid_frac=0, corners=load_r3_corners(f))
    assert [r.record_time for r in recs] == [r.record_time for r in official]
    assert all(np.allclose(r.corners, quad) for r in recs)
    # through the config, like training does
    cfg = load_config(ROOT / "configs" / "r3.yaml",
                      [f"data.root={root.as_posix()}", f"data.r3_corners={f.as_posix()}"])
    from pianovam_vision.metadata import recordings_from_cfg
    assert all(np.allclose(r.corners, quad) for r in recordings_from_cfg(cfg))
    # one session missing -> error naming it (never a silent mix with official boxes)
    f.write_text(json.dumps({"format": "r3_corners_v1", "corners": {s: quad for s in sessions[1:]}}))
    with pytest.raises(KeyError, match=sessions[0]):
        load_recordings_r3(root, valid_frac=0, corners=load_r3_corners(f))
    f.write_text(json.dumps({"format": "r3_corners_v1", "corners": {sessions[0]: quad[:3]}}))
    with pytest.raises(ValueError, match="4"):
        load_r3_corners(f)


def test_corner_export_writes_backgrounds_and_official_corners(tmp_path):
    pytest.importorskip("cv2")
    import cv2
    from pianovam_vision.corners import export
    root = _real_tree(tmp_path)
    cfg = load_config(ROOT / "configs" / "r3.yaml",
                      [f"data.root={root.as_posix()}", "data.frame_size=[320, 180]"])
    cfg["data"]["root"] = str(root)
    out = tmp_path / "frames_out"
    rows = export(cfg, out, chunks_per_session=3, per_chunk=4,
                  only=["rach3_2022-05-05_a01", "ls_2024-02-06_a02"])
    meta = json.loads((out / "sessions.json").read_text())
    assert meta["format"] == "r3_sessions_v1" and len(meta["sessions"]) == 2
    by = {r["session"]: r for r in meta["sessions"]}
    from pianovam_vision.metadata import recordings_from_cfg
    rec = next(r for r in recordings_from_cfg(cfg) if r.session == "rach3_2022-05-05_a01")
    assert np.allclose(by["rach3_2022-05-05_a01"]["official"], rec.corners, atol=0.01)
    assert by["rach3_2022-05-05_a01"]["splits"] == ["test"]
    assert {r["group"] for r in rows} == {0, 1}                      # the two subsets differ
    img = cv2.imread(str(out / by["ls_2024-02-06_a02"]["image"]))
    assert img.shape == (180, 320, 3)
    assert by["ls_2024-02-06_a02"]["frame_size"] == [320, 180]


def _synthetic_keyboard(corners):
    """1920x1080 image of a flat 88-key keyboard whose strip corners are ``corners``."""
    cv2 = pytest.importorskip("cv2")
    from pianovam_vision.corners import STRIP_H, STRIP_W, homography64
    from pianovam_vision.keyboard import key_geometry

    w, h = STRIP_W, STRIP_H
    tex = np.full((h, w, 3), 210, np.uint8)
    for j in range(1, 52):
        x = int(round(j * w / 52))
        tex[:, x - 1:x + 1] = 90                                  # white-key gaps
    for kind, x0, y0, x1, y1 in key_geometry(w, h).values():
        if kind == "black":
            tex[y0:y1, x0:x1] = 30
    inv = np.linalg.inv(homography64(corners))
    return cv2.warpPerspective(tex, inv, (1920, 1080), flags=cv2.INTER_LINEAR,
                               borderValue=(120, 120, 120))


def test_refine_straightens_a_bowed_quad():
    """A front corner 6 px off bows the key grid (ends right, middle off); refine pulls
    that session in line with the others, leaves good and kept sessions alone."""
    pytest.importorskip("scipy")
    from pianovam_vision.corners import refine_sessions

    true = np.array([[1500.0, 1000.0], [400.0, 1000.0], [420.0, 860.0], [1480.0, 860.0]])
    img = _synthetic_keyboard(true)
    bowed = true + [[0, 0], [0, 0], [0, 0], [0, -6]]
    corners = {"s0": bowed, "s1": true, "s2": true, "s3": true, "s4": bowed}
    new, rows = refine_sessions({s: img for s in corners}, corners, keep=["s0"])
    by = {r["session"]: r for r in rows}
    worst = lambda v: np.nanmax(np.abs(v))
    assert worst(by["s4"]["before"]) > 0.15                      # the bow is there...
    assert worst(by["s4"]["after"]) < 0.05                       # ...and gone
    assert np.abs(new["s4"][:, 0] - true[:, 0]).max() < 2.0
    assert np.array_equal(new["s0"], bowed) and by["s0"]["kept"]  # kept as set by hand
    for s in ("s1", "s2", "s3"):
        assert by[s]["move_px"] < 0.5 and worst(by[s]["after"]) < 0.05


def test_save_corners_round_trip(tmp_path):
    from pianovam_vision.corners import save_corners
    from pianovam_vision.metadata import load_r3_corners

    c = {"b_session": np.array([[1.234, 2], [3, 4], [5, 6], [7, 8.5]]),
         "a_session": np.zeros((4, 2))}
    save_corners(tmp_path / "c.json", c, about="test", refined={"kept_as_set_by_hand": []})
    back = load_r3_corners(tmp_path / "c.json")
    assert list(back) == ["a_session", "b_session"]
    assert np.allclose(back["b_session"], [[1.23, 2], [3, 4], [5, 6], [7, 8.5]])
    assert json.loads((tmp_path / "c.json").read_text())["about"] == "test"
