"""Model size, compute and speed (the efficiency numbers of paper 1).

    python -m pianovam_vision.speed --config configs/vam_full.yaml                   # KeyWise
    python -m pianovam_vision.speed --config configs/vam_full.yaml model.arch=strip   # global variant
    python -m pianovam_vision.speed --config configs/vam_global88.yaml                # Global-88
    python -m pianovam_vision.speed --config configs/vam_full.yaml \
        --checkpoint runs/vam_full_s2/best.pt --record_time 2024-02-14_19-55-17 --json results/speed_keywise.json

Reports
  * parameters;
  * multiply-accumulate operations (MACs) per video frame, counted with forward hooks on every
    Conv1d / Conv2d / Linear / GRU (BatchNorm, ReLU, pooling, bias additions and the sigmoids
    are left out, as usual), per part of the model;
  * model-only speed for one window of --frames frames (512 = what inference uses), in float32
    (what evaluate/infer use) and, on a GPU, with fp16 autocast, as frames per second, with
    the peak GPU memory of each pass;
  * with --record_time: the end-to-end time for the first --seconds of a test recording read
    from the video, not from the strip cache (decode + perspective warp + model + decoding to
    notes), and the reading alone, as multiples of real time.
The weights do not change the speed, so --checkpoint is optional (random weights otherwise;
the end-to-end note decoding is then not representative).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .config import config_from_checkpoint, load_config
from .model import build_model


def count_macs(model: nn.Module, x: torch.Tensor) -> Dict[str, float]:
    """MACs of one forward pass of ``x`` (B,T,C,H,W), per frame, by top-level module."""
    macs: Dict[str, float] = defaultdict(float)
    hooks = []

    def part(name: str) -> str:
        return name.split(".")[0] if "." in name else name

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            def f(m, inp, out, key=part(name)):
                macs[key] += out.numel() * (m.in_channels // m.groups) * int(np.prod(m.kernel_size))
        elif isinstance(mod, nn.Linear):
            def f(m, inp, out, key=part(name)):
                macs[key] += out.numel() * m.in_features
        elif isinstance(mod, nn.GRU):
            def f(m, inp, out, key=part(name)):
                s = inp[0].shape
                n_seq, steps = (s[0], s[1]) if m.batch_first else (s[1], s[0])
                d = 2 if m.bidirectional else 1
                per_step = sum(d * 3 * m.hidden_size * ((m.input_size if layer == 0 else d * m.hidden_size)
                                                        + m.hidden_size) for layer in range(m.num_layers))
                macs[key] += n_seq * steps * per_step
        else:
            continue
        hooks.append(mod.register_forward_hook(f))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    frames = x.shape[0] * x.shape[1]
    return {k: v / frames for k, v in macs.items()}


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def model_speed(model, shape, device: str, fp16: bool, reps: int) -> Dict[str, float]:
    x = torch.rand(*shape, device=device)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=fp16):
        model(x)                                      # warm-up
        sync(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        sync(device)
    dt = (time.perf_counter() - t0) / reps
    out = {"frames": shape[1], "seconds_per_window": dt, "frames_per_second": shape[1] / dt}
    if device == "cuda":
        out["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return out


def end_to_end(model, cfg, record_time: str, seconds: float, device: str) -> Dict[str, float]:
    from .decode import decode_with_cfg
    from .infer import predict_rolls
    from .metadata import index_by_record_time, recordings_from_cfg
    from .strip_cache import open_reader

    cfg["data"]["strip_cache"] = ""                   # read the video itself
    rec = index_by_record_time(recordings_from_cfg(cfg))[record_time]
    fps = cfg["labels"]["fps"]
    max_frames = int(round(seconds * fps)) if seconds > 0 else 0

    reader = open_reader(cfg, rec, max_frames)       # 1. reading alone: decode + warp
    n = len(reader)
    win, hop = cfg["infer"]["window_frames"], cfg["infer"]["window_hop"]
    t0 = time.perf_counter()
    for s in range(0, max(1, n), hop):
        if min(s + win, n) > s:
            reader.read_warped(np.arange(s, min(s + win, n)))
    t_read = time.perf_counter() - t0

    reader = open_reader(cfg, rec, max_frames)       # 2. everything
    sync(device)
    t0 = time.perf_counter()
    onset_p, frame_p, vel_p = predict_rolls(model, reader, cfg, device)
    notes = decode_with_cfg(onset_p, frame_p, vel_p, cfg)
    sync(device)
    t_all = time.perf_counter() - t0
    dur = n / fps
    return {"record": record_time, "video_seconds": dur, "frames": n, "notes": len(notes),
            "read_seconds": t_read, "read_x_realtime": dur / t_read,
            "total_seconds": t_all, "total_x_realtime": dur / t_all}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="optional: build from (and load) this checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frames", type=int, default=512, help="frames per window (inference uses 512)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--record_time", default=None, help="also time a real recording end to end")
    ap.add_argument("--seconds", type=float, default=120.0, help="how much of it (0 = all)")
    ap.add_argument("--json", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = config_from_checkpoint(ckpt, args.config, args.overrides)
    else:
        ckpt, cfg = None, load_config(args.config, args.overrides)
    model = build_model(cfg)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    model = model.to(args.device).eval()
    kb = cfg["keyboard"]
    shape = (1, args.frames, 1 if kb["grayscale"] else 3, kb["warp_height"], kb["warp_width"])

    res: Dict[str, object] = {"arch": cfg["model"].get("arch"), "strip_bins": cfg["model"].get("strip_bins"),
                              "device": args.device, "input": list(shape)}
    if args.device == "cuda":
        res["gpu"] = torch.cuda.get_device_name()
    res["parameters"] = int(sum(p.numel() for p in model.parameters()))
    macs = count_macs(model, torch.rand(1, 8, *shape[2:], device=args.device))
    res["macs_per_frame"] = {k: float(v) for k, v in macs.items()}
    res["macs_per_frame_total"] = float(sum(macs.values()))
    res["macs_per_second_at_fps"] = res["macs_per_frame_total"] * cfg["labels"]["fps"]
    print(f"{res['arch']} (strip_bins={res['strip_bins']}): {res['parameters']:,} parameters")
    print(f"MACs per frame: {res['macs_per_frame_total']:.3e} = "
          + ", ".join(f"{k} {v:.3e}" for k, v in macs.items())
          + f"; per second of video at {cfg['labels']['fps']:g} fps: {res['macs_per_second_at_fps']:.3e}")

    for fp16 in ((False, True) if args.device == "cuda" else (False,)):
        try:
            sp = model_speed(model, shape, args.device, fp16, args.reps)
        except RuntimeError as e:                     # e.g. out of memory
            sp = {"error": str(e).splitlines()[0]}
            if args.device == "cuda":
                torch.cuda.empty_cache()
        res["model_" + ("fp16" if fp16 else "fp32")] = sp
        print(f"model only, {'fp16 autocast' if fp16 else 'float32'}, {args.frames}-frame window on {args.device}: "
              + (f"{sp['frames_per_second']:.0f} frames/s"
                 + (f", peak memory {sp['peak_memory_gb']:.2f} GB" if "peak_memory_gb" in sp else "")
                 if "error" not in sp else sp["error"]))

    if args.record_time:
        e2e = end_to_end(model, cfg, args.record_time, args.seconds, args.device)
        res["end_to_end"] = e2e
        print(f"end to end, {e2e['record']} ({e2e['video_seconds']:.0f} s of video, from the video file): "
              f"{e2e['total_x_realtime']:.1f}x real time ({e2e['total_seconds']:.1f} s); "
              f"reading alone (decode + warp) {e2e['read_x_realtime']:.1f}x real time")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
