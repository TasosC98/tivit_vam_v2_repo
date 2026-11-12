"""Tests for the hand / finger code (KeyWise-hands). No dataset needed.

    cd vision_transcription && python -m pytest tests/test_hands.py -q
"""
import json

import numpy as np
import pytest

from pianovam_vision import hands as H
from pianovam_vision.labels import Note

FPS = 30.0


def test_class_ids():
    assert H.class_id("L", "1") == 0 and H.class_id("L", 5) == 4
    assert H.class_id("R", "1") == 5 and H.class_id("R", "5") == 9
    assert H.class_id("Noinfo", "Noinfo") == H.IGNORE and H.class_id("R", "Noinfo") == H.IGNORE
    for c in range(10):
        assert H.class_id(H.class_hand(c), H.class_finger(c)) == c
    assert H.class_hand(H.IGNORE) == "" and H.class_finger(H.IGNORE) == 0


def _write_fingering(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("onset\tkey_offset\tframe_offset\tnote\tvelocity\thand\tfinger\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def test_read_fingering_keeps_file_order_and_builds_roll(tmp_path):
    p = tmp_path / "rec.tsv"
    _write_fingering(p, [(1.0, 2.0, 3.0, 60, 80, "R", 1),      # long note: capped at label_frames
                         (0.5, 0.6, 0.6, 48, 70, "L", 5),      # short: capped at its key release
                         (1.2, 1.5, 1.5, 62, 70, "Noinfo", "Noinfo")])
    lab = H.read_fingering(p)
    assert [n.pitch for n, _ in lab] == [60, 48, 62]          # file order, not sorted
    assert [c for _, c in lab] == [5, 4, H.IGNORE]
    roll = H.build_hand_roll(lab, num_frames=90, fps=FPS, label_frames=8)
    k60, k48, k62 = 60 - 21, 48 - 21, 62 - 21
    assert (roll[30:38, k60] == 5).all() and roll[38, k60] == H.IGNORE
    assert (roll[15:18, k48] == 4).all() and roll[18, k48] == H.IGNORE   # 0.5 s .. 0.6 s
    assert (roll[:, k62] == H.IGNORE).all()                    # Noinfo is never a target
    assert (roll == H.IGNORE).sum() == 90 * 88 - 8 - 3


def test_strip_homography_and_key_layout():
    corners = np.array([[100, 300], [1800, 290], [1820, 520], [90, 530]], float)
    m = H.strip_homography(corners)
    got = H.apply_homography(m, corners)
    assert np.allclose(got, [[0, 0], [1, 0], [1, 1], [0, 1]], atol=1e-9)
    from pianovam_vision.keyboard import key_geometry

    cx, hw, ln = H.key_layout()
    W, Ht = 1408, 112
    for p, (kind, x0, y0, x1, y1) in key_geometry(W, Ht).items():
        k = p - 21
        assert abs(cx[k] * W - (x0 + x1) / 2) <= 1.0
        assert (kind == "black") == (ln[k] < 1.0)
        assert abs(2 * hw[k] * W - (x1 - x0)) <= 1.0


def _track_with(tip_x, tip_y, hand=1, finger=2, T=60, other_x=0.1):
    """Hand ``hand`` has finger ``finger``'s tip at (tip_x, tip_y); its other
    fingertips sit 1.5 white keys away; the other hand rests far away."""
    pts = np.zeros((T, 2, 21, 3), np.float32)
    present = np.ones((T, 2), bool)
    for j, lm in enumerate(H.FINGERTIPS):
        dx = 0.0 if j + 1 == finger else (j + 1 - finger) * 1.5 / H.N_WHITE
        pts[:, hand, lm, 0], pts[:, hand, lm, 1] = tip_x + dx, tip_y
        pts[:, 1 - hand, lm, 0], pts[:, 1 - hand, lm, 1] = other_x + j / H.N_WHITE, 0.8
    return H.SkeletonTrack(pts, present)


def test_fingertip_rules_find_the_pressing_finger():
    cx, _, _ = H.key_layout()
    note = Note(1.0, 1.4, 61, 80)                              # C#4, a black key
    tr = _track_with(cx[61 - 21], 0.4, hand=1, finger=2, other_x=0.05)
    for roi in (True, False):
        cls, conf = H.decide([note], H.fingertip_scores([note], tr, FPS, 8, roi=roi))
        assert cls[0] == H.class_id("R", 2) and conf[0] > 0.9
    # No hand detected -> no decision.
    tr.present[:] = False
    cls, _ = H.decide([note], H.fingertip_scores([note], tr, FPS, 8))
    assert cls[0] == H.IGNORE


def test_roi_rule_uses_key_length():
    """A tip in front of a black key is on the white key, not the black one."""
    cx, _, _ = H.key_layout()
    k_black = 61 - 21
    tr = _track_with(cx[k_black], 0.9, hand=1, finger=3)       # far in front of C#4's end (0.62)
    s_black = H.fingertip_scores([Note(1.0, 1.4, 61, 80)], tr, FPS, 8, roi=True)[0]
    s_white = H.fingertip_scores([Note(1.0, 1.4, 60, 80)], tr, FPS, 8, roi=True)[0]
    assert s_white.max() > s_black.max()


def test_skeleton_json_tags_are_swapped(tmp_path):
    lm_a = [[0.30, 0.5, 0.0]] * 21          # tagged "Right" in the file
    lm_b = [[0.70, 0.5, 0.0]] * 21          # tagged "Left"
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"0": {"Left": lm_b, "Right": lm_a}, "2": {"Left": None, "Right": lm_a}}))
    raw = H.read_skeleton_json(p, swap_tags=True)
    assert raw.shape == (3, 2, 21, 3)
    assert np.allclose(raw[0, 0, :, 0], 0.30) and np.allclose(raw[0, 1, :, 0], 0.70)
    assert np.isnan(raw[1]).all() and np.isnan(raw[2, 1]).all()
    raw2 = H.read_skeleton_json(p, swap_tags=False)
    assert np.allclose(raw2[0, 0, :, 0], 0.70)
    corners = np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], float)
    tr = H.project_skeleton(raw, corners, (1920, 1080), native_fps=60, fps=30)
    assert len(tr) == 1 and tr.present[0].all()            # 3 native frames -> 1 label frame
    assert np.allclose(tr.pts[0, 0, :, 0], 0.30, atol=1e-6)
    x = tr.model_input(np.array([0]))
    assert x.shape == (1, 2, 6, 4) and np.allclose(x[0, :, :, 3], 1.0)


def test_landmark_alignment_finds_a_delay():
    """Landmarks 4 frames late are detected without labels and undone by shifted()."""
    cx, _, _ = H.key_layout()
    rng = np.random.default_rng(0)
    T, n = 3000, 400
    pts = np.zeros((T, 2, 21, 3), np.float32)
    pts[..., 0] = 0.02                                       # resting far left
    present = np.ones((T, 2), bool)
    notes = []
    for i in range(n):
        f = 5 + 7 * i
        p = int(rng.integers(40, 100))
        notes.append(Note(f / FPS, (f + 3) / FPS, p, 80))
        pts[f + 4:f + 4 + 3, 1, H.FINGERTIPS[1], 0] = cx[p - 21]   # on the key 4 frames late
        pts[f + 4:f + 4 + 3, 1, H.FINGERTIPS[1], 1] = 0.5
    late = H.SkeletonTrack(pts, present)
    dist, hit = H.alignment_curve(late, notes, FPS, H.ALIGN_SHIFTS)
    best = int(H.ALIGN_SHIFTS[np.nanargmin(dist)])
    assert best in (3, 4, 5) and hit.max() == 1.0
    assert H.choose_offset(dist, ref_shift=0) == best           # clearly better: applied
    flat = np.full(len(H.ALIGN_SHIFTS), 0.01)
    flat[3] = 0.009
    assert H.choose_offset(flat, ref_shift=0) == 0              # a flat curve: noise, not applied
    fixed = late.shifted(best)
    d2, _ = H.alignment_curve(fixed, notes, FPS, H.ALIGN_SHIFTS)
    assert int(H.ALIGN_SHIFTS[np.nanargmin(d2)]) == 0
    assert not fixed.present[-best:].any()                      # nothing invented at the end


def test_skeleton_offsets_file_is_applied(tmp_path):
    offs = tmp_path / "offsets.json"
    offs.write_text(json.dumps({"rec_a": -3}))
    cfg = {"data": {"skeleton_offsets": str(offs)}}
    assert H.skeleton_offset(cfg, "rec_a") == -3 and H.skeleton_offset(cfg, "other") == 0
    assert H.skeleton_offset({"data": {}}, "rec_a") == 0


def test_chord_matching_gives_different_fingers():
    notes = [Note(1.0, 1.3, 60, 80), Note(1.01, 1.3, 64, 80), Note(2.0, 2.3, 67, 80)]
    s = np.full((3, 10), -5.0)
    s[0, 6], s[0, 7] = 0.0, -0.5                            # both chord notes prefer R2
    s[1, 6], s[1, 8] = 0.0, -0.2
    s[2, 6] = 0.0
    free, _ = H.decide(notes, s)
    assert list(free) == [6, 6, 6]
    matched, _ = H.decide(notes, s, chord_matching=True)
    assert matched[0] != matched[1] and matched[2] == 6
    assert set(matched[:2]) == {6, 8}                      # the cheaper of the two swaps


def test_pitch_split_and_probs_read_out():
    notes = [Note(0.5, 0.7, 59, 80), Note(0.5, 0.7, 60, 80)]
    cls, _ = H.decide(notes, H.pitch_split(notes))
    assert [H.class_hand(c) for c in cls] == ["L", "R"]
    probs = np.full((40, 88, 10), 0.01, np.float32)
    probs[15:20, 59 - 21, 3] = 0.91                          # L4 while the note is held
    cls, conf = H.decide(notes[:1], H.probs_scores(notes[:1], probs, FPS, 8))
    assert cls[0] == 3 and conf[0] > 0.5


def test_mcnemar():
    from pianovam_vision.evaluate_hands import mcnemar_p

    a = np.array([True] * 20 + [False] * 0)
    b = np.array([True] * 10 + [False] * 10)
    assert mcnemar_p(a, b) == pytest.approx(2 / 2 ** 10)
    assert mcnemar_p(a, a) == 1.0


# ------------------------------------------------------------------ model (torch)
torch = pytest.importorskip("torch")


def _cfg(hands=True, skeleton=True, video=True):
    cfg = {
        "keyboard": {"grayscale": False},
        "model": {"arch": "tiled", "encoder_channels": [4, 8, 8, 8], "feature_dim": 16,
                  "gru_hidden": 8, "gru_layers": 1, "dropout": 0.0, "use_velocity": False,
                  "tile_feature_dim": 8, "tile_gru_hidden": 8},
    }
    if hands:
        cfg["model"]["hands"] = {"enable": True, "skeleton": skeleton, "video": video, "dim": 16}
    return cfg


def _inputs(B=2, T=6):
    g = torch.Generator().manual_seed(0)
    frames = torch.rand(B, T, 3, 32, 1408, generator=g)
    skel = torch.zeros(B, T, 2, 6, 4)
    skel[..., 0] = torch.rand(B, T, 2, 6, generator=g)
    skel[..., 1] = 0.7
    skel[..., 3] = 1.0
    return frames, skel


def test_hand_branch_shapes_and_variants():
    from pianovam_vision.model import build_model

    frames, skel = _inputs()
    for kw in ({}, {"skeleton": False}, {"video": False}):
        m = build_model(_cfg(**kw)).eval()
        out = m(frames, skel=skel)
        assert out["hand_logits"].shape == (2, 6, 88, 10)
        assert torch.isfinite(m(frames)["hand_logits"]).all()      # no landmarks given
    with pytest.raises(ValueError):
        build_model(_cfg(skeleton=False, video=False))
    with pytest.raises(ValueError):
        build_model({**_cfg(), "model": {**_cfg()["model"], "arch": "strip"}})


def test_hand_branch_never_changes_the_transcription():
    """Same weights with and without a hand branch -> identical onset/frame
    outputs, and a transcription checkpoint loads into a hand model (only the
    hand branch is missing)."""
    from pianovam_vision.model import build_model

    torch.manual_seed(0)
    plain = build_model(_cfg(hands=False)).eval()
    withh = build_model(_cfg()).eval()
    missing, unexpected = withh.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("hand_branch.") for k in missing)
    frames, skel = _inputs()
    a, b = plain(frames), withh(frames, skel=skel)
    for k in ("onset_logits", "frame_logits"):
        assert torch.equal(a[k], b[k])


def test_skeleton_key_features_point_at_the_key():
    from pianovam_vision.model import SkeletonKeyFeatures

    f = SkeletonKeyFeatures()
    cx, _, _ = H.key_layout()
    skel = torch.zeros(1, 1, 2, 6, 4)
    skel[0, 0, 1, 2] = torch.tensor([cx[39], 0.8, 0.0, 1.0])     # right index tip on C4 (white)
    out = f(skel).reshape(1, 1, 88, 2, 6, SkeletonKeyFeatures.N_FEATS)
    inside = out[0, 0, :, 1, 2, 3]
    assert inside[39] == 1 and inside.sum() == 1
    assert out[0, 0, :, 0].abs().sum() == 0                       # left hand absent -> zeros


def test_frozen_training_changes_only_the_hand_branch():
    from pianovam_vision.model import build_model
    from pianovam_vision.train import (BatchNormGuard, freeze_transcription, hand_loss,
                                       set_train_mode)

    torch.manual_seed(0)
    m = build_model(_cfg())
    freeze_transcription(m)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    frames, skel = _inputs()
    batch = {"hand": torch.randint(-1, 10, (2, 6, 88))}
    for _ in range(3):
        set_train_mode(m, frozen=True)
        out = m(frames, skel=skel)
        loss = hand_loss(out, batch, "cpu")
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = m.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed and all(k.startswith("hand_branch.") for k in changed)
    assert BatchNormGuard(m).finite()
