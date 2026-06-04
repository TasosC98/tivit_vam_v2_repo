"""Pure-numpy tests for label construction and decoding (no torch/video needed).

    cd vision_transcription && python -m pytest tests/ -q
"""
import numpy as np

from pianovam_vision.labels import Note, build_target_rolls
from pianovam_vision.decode import decode_notes


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
