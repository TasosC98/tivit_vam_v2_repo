"""offset_scan finds a time lag, a pitch shift (with a lag) and drift; CPU only.

    cd vision_transcription && python -m pytest tests/test_offset_scan.py -q
"""
import numpy as np

from pianovam_vision.labels import Note
from pianovam_vision.offset_scan import scan, short_share

LAGS = np.arange(-2000, 2001, 20) / 1000.0


def notes(n=600, seed=0, length=120.0):
    rng = np.random.default_rng(seed)
    on = np.sort(rng.uniform(0, length, n))
    return [Note(float(t), float(t) + 0.2, int(p), 64) for t, p in zip(on, rng.integers(30, 90, n))]


def moved(ns, dt=0.0, dp=0, rate=1.0):
    return [Note(n.onset * rate + dt, n.offset * rate + dt, n.pitch + dp, n.velocity) for n in ns]


def test_perfect_prediction():
    ref = notes()
    r = scan(ref, ref, 0.05, LAGS, 4)
    assert r["F1_0"] > 0.99 and r["lag_ms"] == 0 and r["shift"] == 0


def test_late_predictions_give_a_negative_lag():
    ref = notes()
    r = scan(ref, moved(ref, dt=0.3), 0.05, LAGS, 4)
    assert r["F1_0"] < 0.3 and abs(r["lag_ms"] + 300) <= 1e-6 and r["F1_lag"] > 0.99
    assert abs(r["lag_first_ms"] - r["lag_last_ms"]) <= 20                  # constant: no drift


def test_pitch_shift_with_a_time_offset_is_found():
    ref = notes()
    r = scan(ref, moved(ref, dt=-0.5, dp=-2), 0.05, LAGS, 4)
    assert r["F1_lag"] < 0.3                                                 # time alone fails
    assert r["shift"] == 2 and abs(r["shift_lag_ms"] - 500) <= 1e-6 and r["F1_shift"] > 0.99


def test_frame_rate_error_shows_as_drift():
    ref = notes(length=300.0)
    r = scan(ref, moved(ref, rate=1.004), 0.05, LAGS, 0)                    # 0.4 % slow clock
    assert r["lag_first_ms"] > r["lag_last_ms"] + 600                        # ~ -0.2 s vs ~ -1.0 s


def test_short_share():
    ns = [Note(0.0, 0.01, 60, 64), Note(1.0, 1.5, 60, 64)]
    assert short_share(ns, 30.0) == 0.5


def test_octave_transposition_and_mirrored_keyboard():
    ref = notes()
    r = scan(ref, moved(ref, dp=-12), 0.05, LAGS, 12)
    assert r["F1_0"] < 0.3 and r["shift"] == 12 and r["F1_shift"] > 0.99
    mirrored = [Note(n.onset, n.offset, 129 - n.pitch, n.velocity) for n in ref]
    r = scan(ref, mirrored, 0.05, LAGS, 2)
    assert r["F1_shift"] < 0.3 and r["F1_mirror"] > 0.99
