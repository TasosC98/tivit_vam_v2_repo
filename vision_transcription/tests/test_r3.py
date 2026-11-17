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
    for name, change in (("r3_vel", {"use_velocity": True}),
                         ("r3_global88", {"arch": "strip", "strip_bins": 88})):
        c = load_yaml(ROOT / "configs" / f"{name}.yaml")
        assert c["model"] == dict(r3["model"], **change) and c["data"] == r3["data"]
    ft = load_yaml(ROOT / "configs" / "r3_finetune.yaml")["train"]
    assert ft["init_from"].endswith("best.pt") and ft["lr"] < r3["train"]["lr"]
    ns = load_yaml(ROOT / "configs" / "r3_nosync.yaml")["data"]["exclude_records"]
    assert len(ns) == 10 and all(x.startswith("ls_2024-02-06_a0") for x in ns)


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
