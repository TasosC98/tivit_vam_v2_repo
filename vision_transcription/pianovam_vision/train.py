"""Training entry point.

    python -m pianovam_vision.train --config configs/default.yaml [k.v=...]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

# Reduce CUDA fragmentation OOMs (must be set before torch initialises CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Some PianoVAM .mp4 files are slow to seek near EOF; raise decord's retry limit
# so reading the last frames of those videos doesn't crash the run.
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "40960")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .dataset import ClipDataset
from .metadata import filter_by_split, load_recordings
from .metrics import frame_prf
from .model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(cfg: Dict[str, Any]):
    root = Path(cfg["data"]["root"])
    recs = load_recordings(root / cfg["data"]["metadata"])
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


def compute_loss(out, batch, cfg, device):
    t = cfg["train"]
    onset_w = torch.tensor(t["onset_pos_weight"], device=device)
    frame_w = torch.tensor(t["frame_pos_weight"], device=device)
    bce_onset = nn.BCEWithLogitsLoss(pos_weight=onset_w)
    bce_frame = nn.BCEWithLogitsLoss(pos_weight=frame_w)

    onset_t = batch["onset"].to(device)
    frame_t = batch["frame"].to(device)
    loss = bce_onset(out["onset_logits"], onset_t) + bce_frame(out["frame_logits"], frame_t)

    if cfg["model"]["use_velocity"] and "velocity" in out:
        vel_t = batch["velocity"].to(device)
        mask = onset_t  # only supervise velocity where a note starts
        mse = ((out["velocity"] - vel_t) ** 2 * mask).sum() / (mask.sum() + 1e-6)
        loss = loss + t["velocity_loss_weight"] * mse
    return loss


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
        return {"onset_f1": 0.0, "frame_f1": 0.0}
    op = np.concatenate([a.reshape(-1, a.shape[-1]) for a in onset_p])
    fp = np.concatenate([a.reshape(-1, a.shape[-1]) for a in frame_p])
    ot = np.concatenate([a.reshape(-1, a.shape[-1]) for a in onset_t])
    ft = np.concatenate([a.reshape(-1, a.shape[-1]) for a in frame_t])
    of = frame_prf(op, ot, cfg["decode"]["onset_threshold"])
    ff = frame_prf(fp, ft, cfg["decode"]["frame_threshold"])
    return {"onset_f1": of["f1"], "frame_f1": ff["f1"],
            "onset_p": of["precision"], "onset_r": of["recall"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
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

    best_f1 = -1.0
    for epoch in range(t["epochs"]):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for step, batch in enumerate(pbar):
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
                pbar.set_postfix(loss=f"{running / (step + 1):.4f}")

        if (epoch + 1) % t["eval_every_epochs"] == 0:
            metrics = evaluate(model, valid_loader, cfg, device)
            print(f"[epoch {epoch}] valid {metrics}")
            ckpt = {"model": model.state_dict(), "cfg": cfg,
                    "epoch": epoch, "metrics": metrics}
            torch.save(ckpt, out_dir / "last.pt")
            if metrics["frame_f1"] > best_f1:
                best_f1 = metrics["frame_f1"]
                torch.save(ckpt, out_dir / "best.pt")
                print(f"  -> new best frame_f1={best_f1:.4f} (saved best.pt)")

    print(f"done. best frame_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
