"""Pure-numpy tests for label construction and decoding (no torch/video needed).

    cd vision_transcription && python -m pytest tests/ -q
"""
import numpy as np

from pianovam_vision.labels import Note, build_target_rolls
from pianovam_vision.decode import decode_notes
from pianovam_vision.metadata import load_recordings_pianoyt


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
