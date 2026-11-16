"""scripts/extract_landmarks.py without MediaPipe: tag assignment, the PianoVAM
JSON layout (read back by hands.read_skeleton_json) and the comparison."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from pianovam_vision import hands

_spec = importlib.util.spec_from_file_location(
    "extract_landmarks", Path(__file__).resolve().parents[1] / "scripts" / "extract_landmarks.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)


def _hand(x0, y0):
    return [[x0 + 0.01 * i, y0 + 0.005 * i, -0.001 * i] for i in range(21)]


def _write(path, n=40, swap=False, shift=0, drop_right=()):
    data = {}
    for i in range(n):
        j = i + shift
        left, right = _hand(0.2 + 0.002 * j, 0.6), _hand(0.6 - 0.002 * j, 0.55)
        if swap:
            left, right = right, left
        data[str(i)] = {"Left": left, "Right": None if i in drop_right else right}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(path)


def test_assign_tags_resolves_duplicate_labels():
    out, k = E._assign_tags([("Right", 0.7, "b"), ("Right", 0.9, "a")])
    assert out == {"Left": "b", "Right": "a"} and k == 1
    out, k = E._assign_tags([("Left", 0.6, "a"), ("Right", 0.8, "b")])
    assert out == {"Left": "a", "Right": "b"} and k == 0


def test_layout_is_read_by_the_hand_code(tmp_path):
    p = _write(tmp_path / "rec.json", n=5, drop_right={2})
    raw = hands.read_skeleton_json(p, swap_tags=True)       # hand 0 = pianist's left = tag "Right"
    assert raw.shape == (5, 2, 21, 3)
    assert np.isnan(raw[2, 0]).all() and not np.isnan(raw[2, 1]).any()
    assert np.allclose(E.read_raw(p)[:, 0], raw[:, 1])


def test_compare_identical_shifted_and_swapped(tmp_path):
    ref = _write(tmp_path / "ref.json")
    r = E.compare(_write(tmp_path / "same.json"), ref)
    assert r["Left"]["median_px"] == 0 and r["tag_swaps"] == 0 and r["best_shift"] == 0
    assert E.verdict([r]).startswith("REPRODUCES")
    r = E.compare(_write(tmp_path / "late.json", shift=-3), ref)   # our frame t+3 = ref frame t
    assert r["best_shift"] == 3 and E.verdict([r]).startswith("differs")
    r = E.compare(_write(tmp_path / "swap.json", swap=True), ref)
    assert r["tag_swaps"] == 1.0 and E.verdict([r]).startswith("differs")
