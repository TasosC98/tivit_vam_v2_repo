"""The whole pipeline on a tiny synthetic dataset, through the command-line tools (CPU, ~1 min).

A fake PianoVAM (metadata_v2.json + TSV labels) and a fake PianoYT (pianoyt.csv + MIDI labels)
are written with a strip cache in place of the videos, then every step of the paper is run:
cache check, training, threshold calibration on valid, evaluation on test, scoring of MIDI
folders, transcription, speed, fine-tuning PianoVAM -> PianoYT and the zero-shot evaluation.
Needs torch, OpenCV, mir_eval, pretty_midi and scipy (no video decoder).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

for mod in ("torch", "cv2", "mir_eval", "pretty_midi", "scipy"):
    pytest.importorskip(mod)

from pianovam_vision.config import load_config  # noqa: E402
from pianovam_vision.labels import Note  # noqa: E402
from pianovam_vision.metadata import recordings_from_cfg  # noqa: E402
from pianovam_vision.midi_io import write_midi  # noqa: E402
from pianovam_vision.strip_cache import expected_meta, write_record_cache  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FRAMES, W, H = 150, 176, 16                      # 5 s per recording, 2 px per key
SMALL = ["keyboard.warp_width=176", "keyboard.warp_height=16",
         "model.encoder_channels=[8,8,8,8]", "model.tile_feature_dim=8", "model.tile_gru_hidden=8",
         "train.epochs=2", "train.clip_len=16", "train.clip_hop=16", "train.batch_size=2",
         "train.num_workers=0", "train.amp=false", "train.device=cpu",
         "infer.window_frames=64", "infer.window_hop=48"]


def run(*args, ok=True):
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONUNBUFFERED="1")
    p = subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True)
    if ok:
        assert p.returncode == 0, f"{args}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
    return p


def fake_notes(seed):
    rng = np.random.default_rng(seed)
    notes = []
    for i in range(40):
        on = 0.1 + i * 0.11
        notes.append(Note(round(on, 3), round(on + float(rng.uniform(0.1, 0.4)), 3),
                          int(rng.integers(30, 100)), int(rng.integers(30, 120))))
    return sorted(notes, key=lambda n: (n.onset, n.pitch))


def strips_for(notes):
    """Dark keyboard with a bright column where a key is held (so the model can learn a little)."""
    s = np.full((FRAMES, H, W, 3), 40, np.uint8)
    for n in notes:
        k = n.pitch - 21
        s[int(round(n.onset * 30)):int(round(n.offset * 30)), :, 2 * k:2 * k + 2] = 220
    return s


def write_yaml(path, base, data):
    path.write_text(f"base: {json.dumps(str(base))}\n" + "data:\n" +
                    "".join(f"  {k}: {json.dumps(v)}\n" for k, v in data.items()), encoding="utf-8")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    corners = "100, 400", "1800, 400", "1800, 560", "100, 560"
    # --- fake PianoVAM
    vam = tmp / "PianoVAM"
    (vam / "TSV").mkdir(parents=True)
    meta = {}
    for i, split in enumerate(["train", "train", "ext-train", "valid", "test", "test"]):
        rt = f"2024-01-0{i + 1}_10-00-00"
        meta[str(i)] = {"record_time": rt, "split": split, **dict(zip(
            ("Point_LT", "Point_RT", "Point_RB", "Point_LB"), corners))}
        with open(vam / "TSV" / f"{rt}.tsv", "w", encoding="utf-8") as f:
            f.write("#onset\tkey_offset\tframe_offset\tnote\tvelocity\n")
            for n in fake_notes(i):
                f.write(f"{n.onset}\t{n.offset}\t{n.offset + 0.2}\t{n.pitch}\t{n.velocity}\n")
    (vam / "metadata_v2.json").write_text(json.dumps(meta), encoding="utf-8")
    vam_cfg = tmp / "vam.yaml"
    write_yaml(vam_cfg, REPO / "configs" / "pianovam.yaml",
               {"root": str(vam), "strip_cache": str(tmp / "cache_vam"), "exclude_records": []})
    # --- fake PianoYT (ids 1..12 train, 20..21 test; valid_frac 0.1 holds one out)
    yt = tmp / "PianoYT"
    (yt / "pianoyt_MIDI").mkdir(parents=True)
    rows = []
    for vid, code in [(i, 1) for i in range(1, 13)] + [(20, 3), (21, 3)]:
        rows.append(f"{vid},'https://youtube.com/watch?v=x{vid}',{code},600,900,40,1880")
        write_midi(fake_notes(100 + vid), yt / "pianoyt_MIDI" / f"audio_{vid}.0.midi")
    (yt / "pianoyt.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    yt_cfg = tmp / "yt.yaml"
    write_yaml(yt_cfg, REPO / "configs" / "pianoyt.yaml",
               {"root": str(yt), "strip_cache": str(tmp / "cache_yt"), "exclude_records": []})
    # --- strip caches in place of the videos
    for cfg_path in (vam_cfg, yt_cfg):
        cfg = load_config(cfg_path, SMALL)
        from pianovam_vision.labels import read_reference
        for rec in recordings_from_cfg(cfg):
            write_record_cache(strips_for(read_reference(rec, cfg)), cfg["data"]["strip_cache"],
                               rec.record_time, expected_meta(cfg, rec))
    return tmp, vam_cfg, yt_cfg


def test_pianovam_pipeline(world):
    tmp, cfg, _ = world
    run("-m", "pianovam_vision.build_cache", "--config", str(cfg), "--check", *SMALL)
    out = tmp / "runs" / "vam"
    p = run("-m", "pianovam_vision.train", "--config", str(cfg), *SMALL, f"train.out_dir={out}")
    assert "new best onset_f1_best" in p.stdout and (out / "best.pt").exists() and (out / "last.pt").exists()
    run("-m", "pianovam_vision.train", "--config", str(cfg), "--resume", "auto", *SMALL,
        "train.epochs=3", f"train.out_dir={out}")                      # resumes at epoch 2

    p = run("-m", "pianovam_vision.calibrate", "--config", str(cfg), "--checkpoint", str(out / "best.pt"),
            "--data_config", str(cfg), "--split", "valid", "--target", "full_f1",
            "--rolls_dir", str(tmp / "rolls" / "valid"))
    decode = [ln.strip() for ln in p.stdout.splitlines() if ln.strip().startswith("decode.onset_threshold=")]
    assert len(decode) == 1
    p = run("-m", "pianovam_vision.evaluate", "--config", str(cfg), "--checkpoint", str(out / "best.pt"),
            "--data_config", str(cfg), "--split", "test", "--csv", str(tmp / "test.csv"),
            "--save_midi", str(tmp / "midi"), "--rolls_dir", str(tmp / "rolls" / "test"), *decode[0].split())
    assert "headline: Note F1" in p.stdout and len(list((tmp / "midi").glob("*.mid"))) == 2
    rows = (tmp / "test.csv").read_text().splitlines()
    assert len(rows) == 3 and "offtol_f1@100" in rows[0] and "frame_f1" in rows[0]

    p = run("-m", "pianovam_vision.score_midi", "--config", str(cfg), "--split", "test",
            "--pred", str(tmp / "midi"), "--baseline", str(tmp / "midi"), "--csv", str(tmp / "vs.csv"))
    assert "Wilcoxon" in p.stdout and (tmp / "vs.csv").exists()
    rec = "2024-01-05_10-00-00"
    run("-m", "pianovam_vision.infer", "--config", str(cfg), "--checkpoint", str(out / "best.pt"),
        "--data_config", str(cfg), "--record_time", rec, "--output", str(tmp / "one.mid"), *decode[0].split())
    assert (tmp / "one.mid").exists()
    p = run("-m", "pianovam_vision.speed", "--config", str(cfg), "--frames", "16", "--reps", "1", *SMALL)
    assert "parameters" in p.stdout


def test_velocity_and_global_variants_train(world):
    tmp, cfg, _ = world
    for name, extra in (("vel", ["model.use_velocity=true"]),
                        ("glob", ["model.arch=strip", "model.strip_bins=88", "model.feature_dim=16",
                                  "model.gru_hidden=8"])):
        out = tmp / "runs" / name
        run("-m", "pianovam_vision.train", "--config", str(cfg), *SMALL, *extra, "train.epochs=1",
            f"train.out_dir={out}")
        p = run("-m", "pianovam_vision.evaluate", "--config", str(cfg), "--checkpoint", str(out / "best.pt"),
                "--data_config", str(cfg), "--split", "test")
        assert ("++Vel" in p.stdout) == (name == "vel")


def test_pianoyt_finetune_and_zero_shot(world):
    tmp, vam_cfg, yt_cfg = world
    vam = tmp / "runs" / "vam_for_yt"
    run("-m", "pianovam_vision.train", "--config", str(vam_cfg), *SMALL, "train.epochs=1", f"train.out_dir={vam}")
    ft = tmp / "runs" / "yt_ft"
    p = run("-m", "pianovam_vision.train", "--config", str(yt_cfg), *SMALL, "train.epochs=1",
            f"train.init_from={vam / 'best.pt'}", "train.lr=0.0002", f"train.out_dir={ft}")
    assert "initialised weights from" in p.stdout and "missing=[] unexpected=[]" in p.stdout
    assert "train recordings=11" in p.stdout and "valid recordings=1" in p.stdout   # 12 train ids, 1 held out
    p = run("-m", "pianovam_vision.calibrate", "--config", str(vam_cfg), "--checkpoint", str(vam / "best.pt"),
            "--data_config", str(yt_cfg), "--split", "valid", "--target", "onset_f1")
    decode = [ln.strip() for ln in p.stdout.splitlines() if ln.strip().startswith("decode.onset_threshold=")]
    p = run("-m", "pianovam_vision.evaluate", "--config", str(vam_cfg), "--checkpoint", str(vam / "best.pt"),
            "--data_config", str(yt_cfg), "--split", "test", *decode[0].split())
    assert "pianoyt:test (2 recordings)" in p.stdout
