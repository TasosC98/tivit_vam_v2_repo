"""Onset timing options of the decoder, and the probability-roll cache.

Needs only numpy (no torch), so it also runs on a laptop:
    python -m pytest tests/test_decode_timing.py
"""
import json

import numpy as np
import pytest

from pianovam_vision.decode import decode_notes, decode_with_cfg, describe_timing, parse_grid
from pianovam_vision.rolls import RollsCache

FPS = 30.0
C4, CS4 = 60, 61          # a white and a black key
K_C4, K_CS4 = C4 - 21, CS4 - 21


def rolls(T=40):
    return np.zeros((T, 88)), np.zeros((T, 88))


def test_defaults_are_the_classic_rule():
    on, fr = rolls()
    on[10:12, K_C4] = 0.95
    fr[10:20, K_C4] = 0.9
    (n,) = decode_notes(on, fr, FPS, 0.5, 0.5)
    assert (n.pitch, n.onset, n.offset) == (C4, 10 / FPS, 20 / FPS)
    cfg = {"labels": {"fps": FPS}, "decode": {"onset_threshold": 0.5, "frame_threshold": 0.5,
                                              "min_duration_s": 0.03, "default_velocity": 80}}
    assert decode_with_cfg(on, fr, None, cfg) == [n]
    assert describe_timing(cfg["decode"]) == ""


def test_centroid_is_subframe_and_threshold_independent():
    on, fr = rolls()
    fr[10:20, K_C4] = 0.9
    on[9:13, K_C4] = [0.2, 0.6, 0.8, 0.2]        # peak leans towards frame 11
    first = {thr: decode_notes(on, fr, FPS, thr, 0.5)[0].onset * FPS for thr in (0.5, 0.7)}
    assert first[0.5] == pytest.approx(10) and first[0.7] == pytest.approx(11)   # a whole frame apart
    cen = {thr: decode_notes(on, fr, FPS, thr, 0.5, onset_timing="centroid")[0].onset * FPS
           for thr in (0.5, 0.7)}
    expected = (9 * 0.2 + 10 * 0.6 + 11 * 0.8 + 12 * 0.2) / 1.8
    assert cen[0.5] == pytest.approx(expected) and cen[0.7] == pytest.approx(expected)


def test_centroid_ignores_a_neighbouring_note_on_the_same_key():
    on, fr = rolls()
    fr[5:30, K_C4] = 0.9
    on[9:13, K_C4] = [0.3, 0.9, 0.9, 0.3]        # note 1, centred at 10.5
    on[13:16, K_C4] = [0.1, 0.9, 0.9]             # note 2 starts right after the dip
    a, b = decode_notes(on, fr, FPS, 0.5, 0.5, onset_timing="centroid")
    assert a.onset * FPS == pytest.approx(10.5)
    assert b.onset * FPS > 14


def test_time_and_black_key_shifts():
    on, fr = rolls()
    for k in (K_C4, K_CS4):
        on[10:12, k] = 0.95
        fr[10:20, k] = 0.9
    notes = {n.pitch: n for n in decode_notes(on, fr, FPS, 0.5, 0.5,
                                              time_shift_s=-0.01, black_shift_s=-0.02)}
    assert notes[C4].onset == pytest.approx(10 / FPS - 0.01)
    assert notes[CS4].onset == pytest.approx(10 / FPS - 0.03)
    assert notes[CS4].offset == pytest.approx(20 / FPS - 0.03)


def test_times_never_negative_or_inverted():
    on, fr = rolls()
    on[0:2, K_C4] = 0.95
    fr[0:3, K_C4] = 0.9
    (n,) = decode_notes(on, fr, FPS, 0.5, 0.5, onset_timing="centroid", time_shift_s=-0.2)
    assert n.onset == 0.0 and n.offset >= n.onset


def test_timing_never_changes_which_notes_exist():
    rng = np.random.default_rng(0)
    on, fr = rng.random((400, 88)) ** 4, rng.random((400, 88))
    base = [(n.pitch, round(n.onset * FPS)) for n in decode_notes(on, fr, FPS, 0.6, 0.5)]
    for mode in ("first", "centroid"):
        for shift in (-0.03, 0.0, 0.11):
            got = decode_notes(on, fr, FPS, 0.6, 0.5, onset_timing=mode,
                               time_shift_s=shift, black_shift_s=-0.01)
            assert len(got) == len(base)
    with pytest.raises(ValueError):
        decode_notes(on, fr, FPS, onset_timing="peak")


def test_parse_grid():
    assert parse_grid("-0.02:0.02:0.01") == [-0.02, -0.01, 0.0, 0.01, 0.02]
    assert parse_grid("0.1, 0,0.1") == [0.0, 0.1]
    assert 0.0 in parse_grid("-0.04:0.04:0.01")        # the baseline is always tried
    with pytest.raises(ValueError):
        parse_grid("0.1:0:0.01")


def test_rolls_cache_roundtrip_and_staleness(tmp_path):
    meta = {"checkpoint": "a.pt", "split": "valid"}
    onset, frame = np.random.default_rng(1).random((2, 50, 88))
    c = RollsCache(str(tmp_path), meta)
    assert c.load("rec") is None
    c.save("rec", onset, frame)
    got = RollsCache(str(tmp_path), meta).load("rec")
    assert np.allclose(got[0], onset, atol=1e-6) and np.allclose(got[1], frame, atol=1e-6)
    # Other settings: the old rolls must not be reused.
    assert RollsCache(str(tmp_path), {**meta, "checkpoint": "b.pt"}).load("rec") is None
    assert json.loads((tmp_path / "meta.json").read_text())["checkpoint"] == "b.pt"
    assert RollsCache(None, meta).load("rec") is None
