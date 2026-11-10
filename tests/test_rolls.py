"""Decoding from the config, and the probability-roll cache (numpy only)."""
import json

import numpy as np

from pianovam_vision.decode import decode_notes, decode_with_cfg
from pianovam_vision.rolls import RollsCache

FPS = 30.0
K_C4 = 60 - 21


def test_decode_with_cfg_uses_the_config_thresholds():
    on, fr = np.zeros((40, 88)), np.zeros((40, 88))
    on[10:12, K_C4] = 0.95
    fr[10:20, K_C4] = 0.9
    (n,) = decode_notes(on, fr, FPS, 0.5, 0.5)
    assert (n.pitch, n.onset, n.offset) == (60, 10 / FPS, 20 / FPS)
    cfg = {"labels": {"fps": FPS}, "decode": {"onset_threshold": 0.5, "frame_threshold": 0.5,
                                              "min_duration_s": 0.03, "default_velocity": 80}}
    assert decode_with_cfg(on, fr, None, cfg) == [n]
    cfg["decode"]["onset_threshold"] = 0.96                      # above the onset peak
    assert decode_with_cfg(on, fr, None, cfg) == []


def test_rolls_cache_roundtrip_and_staleness(tmp_path):
    meta = {"checkpoint": "a.pt", "split": "valid"}
    onset, frame = np.random.default_rng(1).random((2, 50, 88))
    c = RollsCache(str(tmp_path), meta)
    assert c.load("rec") is None
    c.save("rec", onset, frame)
    got = RollsCache(str(tmp_path), meta).load("rec")
    assert np.allclose(got[0], onset, atol=1e-6) and np.allclose(got[1], frame, atol=1e-6)
    assert got[2] is None                               # no velocity head
    c.save("rec_v", onset, frame, frame[::-1])
    assert np.allclose(RollsCache(str(tmp_path), meta).load("rec_v")[2], frame[::-1], atol=1e-6)
    # Other settings: the old rolls must not be reused.
    assert RollsCache(str(tmp_path), {**meta, "checkpoint": "b.pt"}).load("rec") is None
    assert json.loads((tmp_path / "meta.json").read_text())["checkpoint"] == "b.pt"
    assert RollsCache(None, meta).load("rec") is None


def test_rolls_cache_failures_never_raise(tmp_path):
    onset, frame = np.zeros((2, 5, 88))
    c = RollsCache(str(tmp_path / "rolls"), {"x": 1})
    (tmp_path / "rolls" / "bad.npz").write_bytes(b"not a zip file")
    assert c.load("bad") is None                        # unreadable -> recompute
    c.dir = tmp_path / "missing" / "dir"                # saving now fails ...
    c.save("rec", onset, frame)                         # ... without raising
    assert c.dir is None and c.load("rec") is None
