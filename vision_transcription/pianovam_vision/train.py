"""Training entry point.

    python -m pianovam_vision.train --config configs/default.yaml [k.v=...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

# Reduce CUDA fragmentation OOMs (must be set before torch initialises CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Some PianoVAM .mp4 files seek slowly near EOF. Keep decord's retry limit LOW so
# a bad frame fails *fast* and the dataset's skip-a-clip logic handles it, instead
# of decord spinning through tens of thousands of retries (which looks like a hang).
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "2048")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .dataset import ClipDataset
from .metadata import filter_by_split, recordings_from_cfg
from .metrics import frame_prf
from .model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(cfg: Dict[str, Any]):
    recs = recordings_from_cfg(cfg)
    train_recs = filter_by_split(recs, cfg["data"]["train_splits"])
    valid_recs = filter_by_split(recs, cfg["data"]["valid_splits"])

    excl = set(cfg["data"].get("exclude_records", []) or [])
    if excl:
        train_recs = [r for r in train_recs if r.record_time not in excl]
        valid_recs = [r for r in valid_recs if r.record_time not in excl]
        print(f"excluding {len(excl)} record(s): {sorted(excl)}")

    train_ds = ClipDataset(cfg, train_recs, train=True)
    valid_ds = ClipDataset(cfg, valid_recs, train=False)
    print(f"train recordings={len(train_recs)} clips={len(train_ds)} | "
          f"valid recordings={len(valid_recs)} clips={len(valid_ds)}")

    t = cfg["train"]
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=t["batch_size"], shuffle=True,
        num_workers=t["num_workers"], pin_memory=pin, drop_last=True,
        persistent_workers=t["num_workers"] > 0, prefetch_factor=2 if t["num_workers"] > 0 else None,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=t["batch_size"], shuffle=False,
        num_workers=t["num_workers"], pin_memory=pin,
        persistent_workers=t["num_workers"] > 0, prefetch_factor=2 if t["num_workers"] > 0 else None,
    )
    return train_loader, valid_loader


def lr_at(step: int, total_steps: int, t: Dict[str, Any]) -> float:
    """Learning rate at a global step: constant, or linear warmup + cosine decay.

    A pure function of the step, so a resumed run continues the same schedule.
    """
    base = float(t["lr"])
    if t.get("lr_schedule", "constant") != "cosine" or total_steps <= 0:
        return base
    warm = int(total_steps * float(t.get("warmup_frac", 0.03)))
    if step < warm:
        return base * (step + 1) / warm
    prog = min(1.0, (step - warm) / max(1, total_steps - warm))
    floor = float(t.get("min_lr_ratio", 0.02))
    return base * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog)))


def compute_loss(out, batch, cfg, device):
    t = cfg["train"]
    onset_w = torch.tensor(t["onset_pos_weight"], device=device)
    frame_w = torch.tensor(t["frame_pos_weight"], device=device)
    bce_onset = nn.BCEWithLogitsLoss(pos_weight=onset_w)
    bce_frame = nn.BCEWithLogitsLoss(pos_weight=frame_w)

    onset_t = batch["onset"].to(device)
    frame_t = batch["frame"].to(device)
    # Per-head weights: frame_loss_weight=0 trains onsets only (PianoYT, whose
    # MIDI offsets carry sustain-pedal tails the camera cannot see).
    loss = (float(t.get("onset_loss_weight", 1.0)) * bce_onset(out["onset_logits"], onset_t)
            + float(t.get("frame_loss_weight", 1.0)) * bce_frame(out["frame_logits"], frame_t))

    if cfg["model"]["use_velocity"] and "velocity" in out:
        vel_t = batch["velocity"].to(device)
        mask = onset_t  # only supervise velocity where a note starts
        mse = ((out["velocity"] - vel_t) ** 2 * mask).sum() / (mask.sum() + 1e-6)
        loss = loss + t["velocity_loss_weight"] * mse
    return loss


VALID_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SELECT_METRICS = ("frame_f1", "onset_f1", "frame_f1_best", "onset_f1_best")


@torch.no_grad()
def evaluate(model, loader, cfg, device) -> Dict[str, float]:
    model.eval()
    onset_p, frame_p, onset_t, frame_t = [], [], [], []
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        out = model(frames)
        onset_p.append(torch.sigmoid(out["onset_logits"]).cpu().numpy())
        frame_p.append(torch.sigmoid(out["frame_logits"]).cpu().numpy())
        onset_t.append(batch["onset"].numpy())
        frame_t.append(batch["frame"].numpy())
    if not onset_p:
        return {k: 0.0 for k in SELECT_METRICS}
    op = np.concatenate([a.reshape(-1, a.shape[-1]) for a in onset_p])
    fp = np.concatenate([a.reshape(-1, a.shape[-1]) for a in frame_p])
    ot = np.concatenate([a.reshape(-1, a.shape[-1]) for a in onset_t])
    ft = np.concatenate([a.reshape(-1, a.shape[-1]) for a in frame_t])
    of = frame_prf(op, ot, cfg["decode"]["onset_threshold"])
    ff = frame_prf(fp, ft, cfg["decode"]["frame_threshold"])
    out = {"onset_f1": of["f1"], "frame_f1": ff["f1"],
           "onset_p": of["precision"], "onset_r": of["recall"]}
    # Threshold-free variants: best F1 over a threshold grid. Decode thresholds
    # are re-calibrated after training anyway, so these track what will actually
    # be reported better than F1 at a fixed 0.5 (onset_pos_weight inflates the
    # onset probabilities, which moves the best threshold during training).
    for name, p, tgt in (("onset", op, ot), ("frame", fp, ft)):
        f1s = [(frame_prf(p, tgt, thr)["f1"], thr) for thr in VALID_THRESHOLDS]
        best_f1, best_thr = max(f1s)
        out[f"{name}_f1_best"], out[f"{name}_thr_best"] = best_f1, best_thr
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume from, or 'auto' for "
                         "<out_dir>/last.pt if it exists")
    ap.add_argument("overrides", nargs="*", help="dotted overrides key=value")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    t = cfg["train"]
    set_seed(t["seed"])

    # Device comes from the config (usually set by the active server profile):
    # 'auto' -> cuda if present else cpu; 'cpu'/'cuda' force it.
    dev = t.get("device", "auto")
    if dev == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = dev
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: train.device=cuda but CUDA is unavailable; using cpu")
        device = "cpu"
    profile = cfg.get("_active_profile")
    print(f"device={device}" + (f" | profile={profile}" if profile else ""))

    out_dir = Path(t["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    train_loader, valid_loader = make_loaders(cfg)
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model params: {n_params:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"],
                            weight_decay=t["weight_decay"])
    use_amp = bool(t["amp"]) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Validation metric that picks best.pt. frame_* suits PianoVAM (clean
    # key-release labels); onset_* is the fair one for PianoYT, whose frame
    # targets include sustain-pedal tails. *_best = best over a threshold grid.
    select = t.get("select_metric", "frame_f1")
    if select not in SELECT_METRICS:
        raise SystemExit(f"train.select_metric must be one of {SELECT_METRICS}, got {select!r}")
    best_f1 = -1.0
    start_epoch = 0
    resume_path = args.resume
    if resume_path == "auto":
        cand = out_dir / "last.pt"
        resume_path = str(cand) if cand.exists() else None
    if resume_path:
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        best_f1 = ck.get("best_f1", ck.get("metrics", {}).get("frame_f1", -1.0))
        if ck.get("select_metric", "frame_f1") != select:
            best_f1 = -1.0          # best.pt was chosen by another metric
        start_epoch = int(ck.get("epoch", -1)) + 1
        print(f"resumed from {resume_path}: continuing at epoch {start_epoch} "
              f"(best {select}={best_f1:.4f})")
    elif t.get("init_from"):
        # Fine-tuning: start from another run's weights (e.g. PianoVAM ->
        # PianoYT) with a fresh optimizer, schedule and epoch counter.
        ck = torch.load(t["init_from"], map_location=device, weights_only=False)
        src = ck.get("cfg", {})
        for sec, key in (("model", "arch"), ("keyboard", "warp_width"),
                         ("keyboard", "warp_height"), ("keyboard", "grayscale")):
            a, b = src.get(sec, {}).get(key), cfg[sec].get(key)
            if a is not None and a != b:
                print(f"WARNING: init_from was trained with {sec}.{key}={a}, "
                      f"this run uses {b}")
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"initialised weights from {t['init_from']} (its epoch "
              f"{ck.get('epoch', '?')}); missing={list(missing)} "
              f"unexpected={list(unexpected)}")

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * t["epochs"]
    if t.get("lr_schedule", "constant") != "constant":
        print(f"lr schedule: {t['lr_schedule']} (peak {t['lr']}, warmup "
              f"{t.get('warmup_frac', 0.03):.0%} of {total_steps} steps)")

    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        running = 0.0
        t_epoch = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for step, batch in enumerate(pbar):
            lr = lr_at(epoch * steps_per_epoch + step, total_steps, t)
            for g in opt.param_groups:
                g["lr"] = lr
            frames = batch["frames"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(frames)
                loss = compute_loss(out, batch, cfg, device)
            scaler.scale(loss).backward()
            if t["grad_clip"] > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            if step % t["log_every"] == 0:
                pbar.set_postfix(loss=f"{running / (step + 1):.4f}", lr=f"{lr:.2e}")
        print(f"[epoch {epoch}] train loss {running / max(1, steps_per_epoch):.4f} | "
              f"{(time.time() - t_epoch) / 60:.1f} min")

        if (epoch + 1) % t["eval_every_epochs"] == 0:
            metrics = evaluate(model, valid_loader, cfg, device)
            print(f"[epoch {epoch}] valid {metrics}")
            is_best = metrics[select] > best_f1
            if is_best:
                best_f1 = metrics[select]
            # Save optimizer/scaler/best_f1 too so --resume continues cleanly.
            ckpt = {"model": model.state_dict(), "cfg": cfg,
                    "epoch": epoch, "metrics": metrics,
                    "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "best_f1": best_f1, "select_metric": select}
            torch.save(ckpt, out_dir / "last.pt")
            if is_best:
                torch.save(ckpt, out_dir / "best.pt")
                print(f"  -> new best {select}={best_f1:.4f} (saved best.pt)")

    print(f"done. best {select}={best_f1:.4f}")


if __name__ == "__main__":
    main()
