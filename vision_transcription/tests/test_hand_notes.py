"""Tests for the note-level hand/finger path (hand_notes, hand_report, the new
rules and decoders in hands.py). No dataset needed.

    cd vision_transcription && python -m pytest tests/test_hand_notes.py -q
"""
import json

import numpy as np
import pytest

from pianovam_vision import hands as H
from pianovam_vision.labels import Note

FPS = 30.0


def _track_with(tip_x, tip_y, hand=1, finger=2, T=60, other_x=0.1):
    pts = np.zeros((T, 2, 21, 3), np.float32)
    present = np.ones((T, 2), bool)
    for j, lm in enumerate(H.FINGERTIPS):
        dx = 0.0 if j + 1 == finger else (j + 1 - finger) * 1.5 / H.N_WHITE
        pts[:, hand, lm, 0], pts[:, hand, lm, 1] = tip_x + dx, tip_y
        pts[:, 1 - hand, lm, 0], pts[:, 1 - hand, lm, 1] = other_x + j / H.N_WHITE, 0.8
    return H.SkeletonTrack(pts, present)


# ------------------------------------------------------------------ rules / decoders
def test_ayf_rule_picks_the_finger_on_the_key():
    cx, _, _ = H.key_layout()
    note = Note(1.0, 1.4, 64, 80)
    tr = _track_with(cx[64 - 21], 0.5, hand=0, finger=3)
    s = H.ayf_scores([note], tr, FPS)
    p = np.exp(s[0])
    assert p.sum() == pytest.approx(1.0, abs=1e-6)
    cls, _ = H.decide([note], s, joint=True)
    assert cls[0] == H.class_id("L", 3)
    tr.present[:] = False
    assert not np.isfinite(H.ayf_scores([note], tr, FPS)).any()


def test_decide_joint_vs_hand_first():
    # Left hand has more total mass, but the single best class is R1.
    p = np.array([[0.18, 0.18, 0.18, 0.0, 0.0, 0.4, 0.06, 0.0, 0.0, 0.0]])
    notes = [Note(0.0, 0.5, 60, 80)]
    c_hand, _ = H.decide(notes, np.log(p + 1e-12))
    c_joint, _ = H.decide(notes, np.log(p + 1e-12), joint=True)
    assert c_hand[0] < 5 and c_joint[0] == 5


def test_finger_prior_viterbi_uses_the_transitions():
    # Training data: the right hand always plays an ascending scale 1-2-3-4-5.
    notes = [Note(0.2 * i, 0.2 * i + 0.1, 60 + 2 * (i % 5), 80) for i in range(200)]
    y = np.array([5 + i % 5 for i in range(200)])
    prior = H.FingerPrior.fit([(notes, y)])
    assert prior.log_t.shape == (2, 2, 25, 5, 5)
    assert np.allclose(np.exp(prior.log_t).sum(-1), 1.0)
    d = H.PRIOR_MAX_INTERVAL + 2
    assert prior.log_t[1, 0, d, 0].argmax() == 1          # after R1, +2 semitones -> R2
    # Test: the middle note's evidence is ambiguous between R2 and R4; the prior decides.
    t = [Note(0.0, 0.1, 60, 80), Note(0.2, 0.3, 62, 80), Note(0.4, 0.5, 64, 80)]
    pr = np.zeros((3, 10))
    pr[0, 5] = 1.0
    pr[1, 6], pr[1, 8] = 0.45, 0.55
    pr[2, 7] = 1.0
    plain, _ = H.decide(t, np.log(pr + 1e-12))
    assert plain[1] == 8
    assert list(H.viterbi_fingers(t, pr, prior)) == [5, 6, 7]
    # No evidence -> no decision.
    pr[1] = 0.0
    assert H.viterbi_fingers(t, pr, prior)[1] == H.IGNORE
    back = H.FingerPrior.from_dict(json.loads(json.dumps(prior.to_dict())))
    assert np.allclose(back.log_t, prior.log_t)


def test_softmax_rows():
    s = np.array([[0.0, -np.inf] + [-np.inf] * 8, [-np.inf] * 10])
    p = H.softmax_rows(s)
    assert p[0, 0] == 1.0 and p[1].sum() == 0.0


# ------------------------------------------------------------------ note data
from pianovam_vision import hand_notes as HN  # noqa: E402


def _cfg(**hn):
    return {"labels": {"fps": FPS}, "hands": {"label_frames": 8, "native_size": [1920, 1080]},
            "keyboard": {"warp_width": 1408, "warp_height": 112},
            "augment": {"enable": True, "p": 1.0, "hscale": 0.006, "hshift_px": 3, "vshift_px": 3,
                        "rotate_deg": 0.3},
            "model": {"hands": {"dim": 16, "key_kernel": 5, "key_dilations": [1, 2], "time_kernel": 5,
                                "dropout": 0.0}},
            "hand_notes": {"window_notes": 8, "note_dim": 16, "key_embed_dim": 4, "dropout": 0.0, **hn},
            "train": {"device": "cpu"}}


CORNERS = np.array([[120, 350], [1840, 345], [1840, 560], [120, 565]], np.float64)


def _noteset(n=20, T=200, seed=0, **hn):
    rng = np.random.default_rng(seed)
    notes = [Note(0.1 + 0.15 * i, 0.1 + 0.15 * i + 0.3, int(rng.integers(40, 90)), 80) for i in range(n)]
    notes[3] = Note(notes[2].onset, notes[2].offset, notes[2].pitch + 4, 80)      # a chord
    skel = np.zeros((T, 2, 6, 4), np.float32)
    skel[..., 0] = rng.random((T, 2, 6))
    skel[..., 1] = 0.6
    skel[..., 3] = 1.0
    auto = rng.integers(-1, 10, n)
    return HN.make_noteset(notes[::-1], _cfg(**hn), skel, CORNERS, auto=auto[::-1],
                           meta={"record_time": "r"}), notes, auto


def test_noteset_order_readout_and_context():
    ns, notes, auto = _noteset()
    on = [n.onset for n in ns.notes]
    assert on == sorted(on)
    assert list(ns.rows[:2]) == [19, 18]                     # rows point back into the input list
    assert (ns.n_read >= 1).all() and (ns.n_read <= 8).all()
    assert ns.sym.shape == (20, HN.N_SYM)
    i, j = 2, 3                                              # the chord: both rank features set
    assert ns.sym[i, 3] > 0 and ns.sym[j, 3] > 0 and ns.sym[i, 4] != ns.sym[j, 4]
    assert (ns.y == auto).all() and (ns.gt == HN.NO_GT).all()


def test_clip_input_aug_and_image_coordinates():
    ns, _, _ = _noteset()
    frames = np.array([[-1, 0, 1], [198, 199, 200]])
    c = HN.clip_input(ns, frames, "keyrel")
    assert c.shape == (2, 3, 2, 6, 4)
    assert c[0, 0].sum() == 0 and c[1, 2].sum() == 0           # outside the recording: no hands
    rng = np.random.default_rng(0)
    aug = {"p": 1.0, "hscale": 0.006, "hshift_px": 3, "vshift_px": 3, "rotate_deg": 0.3,
           "strip_w": 1408, "strip_h": 112, "corner_jitter": 0.01, "hand_dropout": 0.0}
    a = HN.clip_input(ns, frames, "keyrel", aug, rng)
    assert not np.allclose(a[0, 1, ..., 0], c[0, 1, ..., 0]) and np.abs(a - c).max() < 0.2
    im = HN.clip_input(ns, frames, "image")
    # strip x 0..1 maps between the keyboard's left and right corners in the image
    xs = im[0, 1, :, :, 0]
    assert (xs > 100 / 1920).all() and (xs < 1850 / 1920).all()


@pytest.mark.parametrize("features", ["keyrel", "strip", "image", "none"])
def test_note_model_trains_and_predicts(features):
    torch = pytest.importorskip("torch")
    cfg = _cfg(features=features, steps=3, batch_windows=2, log_every=1)
    sets = [_noteset(seed=s, features=features)[0] for s in range(3)]
    log = HN.Log(None)
    model, prior = HN.train_model(cfg, sets, seed=0, log=log)
    pf, ps = HN.predict_noteset(model, sets[0], "cpu")
    assert ps.shape == (20, 10) and np.allclose(ps.sum(1), 1.0, atol=1e-4)
    if features == "none":
        assert pf is None
    else:
        assert pf.shape == (20, 10) and np.allclose(pf.sum(1), 1.0, atol=1e-4)
    assert isinstance(prior, H.FingerPrior)


def test_residual_note_head_starts_at_the_frame_readout():
    torch = pytest.importorskip("torch")
    cfg = _cfg()
    model = HN.build_note_model(cfg).eval()
    model.hn = cfg["hand_notes"]
    ns = _noteset()[0]
    pf, ps = HN.predict_noteset(model, ns, "cpu")
    assert np.allclose(pf, ps, atol=1e-4)                     # note output = read-out at init


def test_hand_supervision_only_scores_the_hand():
    torch = pytest.importorskip("torch")
    logits = torch.zeros(1, 10)
    logits[0, 5] = 3.0                                        # all right-hand mass on R1
    y = torch.tensor([9])                                     # truth R5
    assert HN.class_nll(logits, y, "hand") < 0.2 < HN.class_nll(logits, y, "finger")
    assert HN.class_nll(logits, torch.tensor([-1]), "finger") == 0.0


def test_folds_never_train_on_what_they_evaluate(tmp_path):
    from pianovam_vision.metadata import Recording

    gt_dir = tmp_path / "Fingering_GT"
    gt_dir.mkdir()
    recs = []
    for i, (date, who) in enumerate([("2024-02-15", "A"), ("2024-02-16", "B"), ("2024-09-02", "A"),
                                     ("2024-09-03", "C"), ("2024-02-17", "C")]):
        rt = f"{date}_10-00-0{i}"
        recs.append(Recording(rt, "train" if i < 3 else "test", CORNERS.astype(np.float32), performer=who))
        if i != 3:
            (gt_dir / f"{rt}.tsv").write_text("x")
    cfg = {"data": {"root": str(tmp_path), "fingering_gt_dir": "Fingering_GT"}, "hand_notes": {}}
    for scheme, n_folds in (("loro", 4), ("lopo", 3), ("session", 2), ("official", 1)):
        folds = HN.make_folds(cfg, scheme, recs)
        assert len(folds) == n_folds, scheme
        for name, tr, ev in folds:
            assert ev and not set(tr) & set(ev)
            if scheme == "lopo":                               # every recording of the pianist is out
                who = {r.performer for r in recs if r.record_time in ev}
                assert not {r.performer for r in recs if r.record_time in tr} & who
            if scheme == "session":
                s = {HN.session_of(cfg, r) for r in ev}
                assert not {HN.session_of(cfg, r) for r in tr} & s


def test_perturbations():
    cx, _, _ = H.key_layout()
    tr = _track_with(cx[40], 0.5, hand=1, finger=2)
    rng = np.random.default_rng(0)
    same = HN.perturb_track(tr, CORNERS, "corner", 0.0, rng)
    assert np.allclose(same.pts, tr.pts, atol=1e-4)
    moved = HN.perturb_track(tr, CORNERS, "corner", 0.01, rng)
    assert not np.allclose(moved.pts[..., 0], tr.pts[..., 0], atol=1e-4)
    sw = HN.perturb_track(tr, CORNERS, "swap", 1.0, rng)
    assert np.allclose(sw.pts[:, 0], tr.pts[:, 1])
    dr = HN.perturb_track(tr, CORNERS, "drop", 1.0, rng)
    assert not dr.present.any() and not dr.pts.any()


def test_e2e_counts():
    ref = [Note(1.0, 1.5, 60, 80), Note(2.0, 2.5, 62, 80), Note(3.0, 3.5, 64, 80)]
    lab = np.array([5, 6, H.IGNORE])
    est = [Note(1.02, 1.5, 60, 80), Note(2.01, 2.5, 62, 80), Note(3.0, 3.5, 64, 80), Note(4.0, 4.5, 65, 80)]
    cls = np.array([5, 7, 7, 5])
    c = HN.e2e_counts(ref, lab, est, cls, 0.05)
    assert c["ref"] == 2 and c["est"] == 3 and c["match"] == 2      # the unlabelled match left out
    assert c["hand"] == 2 and c["finger"] == 1 and c["wrong_finger"] == 1 and c["extra"] == 1


# ------------------------------------------------------------------ report statistics
def test_report_statistics():
    from pianovam_vision import hand_report as R

    rec = np.array(["a"] * 25 + ["b"] * 5)
    assert list(R.block_ids(rec)) == [0] * 10 + [1] * 10 + [2] * 5 + [3] * 5
    v = np.ones(30)
    b = R.bootstrap(v, rec, 50)
    assert b.shape == (50, 1) and np.allclose(b, 1.0)
    assert R.sign_flip_p([1, 1, 1, 1, 1]) == pytest.approx(2 / 32)
    assert R.sign_flip_p([0, 0]) == 1.0


def test_report_end_to_end(tmp_path):
    """Two fake folds -> hand_report runs and finds the perfect method perfect."""
    from pianovam_vision import hand_report as R

    for k, rt in enumerate(("2024-02-15_20-07-54", "2024-09-02_14-10-41")):
        ns, _, _ = _noteset(n=30, seed=k)
        ns.meta.update(record_time=rt, performer=f"P{k}", session="feb" if k == 0 else "sep", split="test")
        ns.gt = np.where(ns.y >= 0, ns.y, 3)
        perfect = np.eye(10)[ns.gt]
        rules = {"s_roi": np.log(np.eye(10)[(ns.gt + 1) % 10] + 1e-9).astype(np.float32),
                 "s_pitch": H.pitch_split(ns.notes).astype(np.float32),
                 "present": np.ones((len(ns), 2), bool)}
        fd = tmp_path / "run" / f"f{k}_s1"
        HN.save_note_npz(fd / "pred" / f"{rt}.npz", ns, {"p_frame": perfect, "p_seq": perfect}, rules,
                         {"seed": 1})
        (fd / "prior.json").write_text(json.dumps(H.FingerPrior.fit([(ns.notes, ns.y)]).to_dict()))
    out = tmp_path / "rep.txt"
    R.main(["--run", f"m={tmp_path / 'run'}", "--out", str(out), "--n_boot", "50",
            "--csv", str(tmp_path / "n.csv"), "--curves", str(tmp_path / "curves")])
    text = out.read_text()
    assert "m:seq" in text and "roi" in text
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("m:seq "))
    assert "100.0" in line
    assert (tmp_path / "curves" / "precision_coverage_manual.csv").exists()
