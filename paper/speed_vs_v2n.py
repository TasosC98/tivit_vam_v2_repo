"""KeyWise and V2N on the same GPU: parameters, MACs (same counting code) and model-only speed.

    python paper/speed_vs_v2n.py --v2n /path/to/V2N --json results/speed_vs_v2n.json

V2N (github.com/yonghyunk1m/V2N) is built from its own code and model config
(configs/oofv_1s/model/default.yaml) with random weights: the speed and the number of
operations do not depend on the weights. V2N needs torchvision (its ResNet-18 blocks); timm,
which V2N's model file imports but this model does not use, is replaced by an empty module
when it is not installed. Both models are timed without video decoding and without note
decoding, in float32 and (on a GPU) with fp16 and bf16 autocast:
  KeyWise  one window of --frames frames at 30 fps (512 = what evaluate/infer use);
  V2N      1-s clips of 25 grayscale 144 x 800 frames (25 fps), --v2n_batch clips at a time.
Both systems process overlapping pieces of the video at test time: KeyWise windows of 512
frames that start every 448 frames (each frame is processed 512/448 = 1.14 times on average),
V2N 1-s clips that start every 0.5 s (each frame twice). The comparison is reported both as
each system runs at test time ("with overlap", the paper's numbers) and without overlap.
MACs are counted with pianovam_vision.speed.count_macs for both (Conv1d/2d/3d, Linear, GRU,
LSTM); V2N reports "125 GFLOPs per 1-s clip", so the per-clip count shows which unit that is.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from pianovam_vision.config import load_config  # noqa: E402
from pianovam_vision.model import build_model  # noqa: E402
from pianovam_vision.speed import count_macs, model_speed  # noqa: E402

V2N_FPS, V2N_CLIP_S, V2N_STRIDE_S = 25.0, 1.0, 0.5


def build_v2n(v2n_dir: str) -> torch.nn.Module:
    sys.path.insert(0, os.path.abspath(v2n_dir))
    try:
        import timm  # noqa: F401
    except ImportError:
        sys.modules["timm"] = types.ModuleType("timm")
    import importlib

    with open(os.path.join(v2n_dir, "configs", "oofv_1s", "model", "default.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def make(spec, **extra):
        mod, cls = spec["_target_"].rsplit(".", 1)
        args = {k: v for k, v in spec.items() if k != "_target_"}
        return getattr(importlib.import_module(mod), cls)(**args, **extra)

    return make(cfg["transcriber"], feature_extractor=make(cfg["feature_extractor"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2n", required=True, help="the cloned V2N repository")
    ap.add_argument("--config", default="configs/pianovam.yaml", help="KeyWise config")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frames", type=int, default=512, help="KeyWise window (frames)")
    ap.add_argument("--v2n_batch", type=int, default=2, help="V2N clips per forward pass (its test batch is 2)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    kcfg = load_config(args.config)
    kb = kcfg["keyboard"]
    k_fps = float(kcfg["labels"]["fps"])
    # How many times each frame is processed on average at test time.
    passes = {"KeyWise": kcfg["infer"]["window_frames"] / kcfg["infer"]["window_hop"],
              "V2N": V2N_CLIP_S / V2N_STRIDE_S}
    systems = {
        "KeyWise": dict(model=build_model(kcfg), fps=k_fps,
                        shape=(1, args.frames, 3, kb["warp_height"], kb["warp_width"])),
        "V2N": dict(model=build_v2n(args.v2n), fps=V2N_FPS,
                    shape=(args.v2n_batch, int(V2N_FPS * V2N_CLIP_S), 144, 800)),
    }
    precisions = {"fp32": None}
    if args.device == "cuda":
        precisions.update(fp16=torch.float16, bf16=torch.bfloat16)
        print("GPU:", torch.cuda.get_device_name())

    res = {"device": args.device, "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
           "passes_per_frame": passes}
    for name, s in systems.items():
        model = s["model"].to(args.device).eval()
        params = sum(p.numel() for p in model.parameters())
        small = (1, min(8, s["shape"][1])) + tuple(s["shape"][2:]) if name == "KeyWise" else (1,) + tuple(s["shape"][1:])
        macs = count_macs(model, torch.rand(*small, device=args.device))
        per_frame = float(sum(macs.values()))
        r = {"parameters": int(params), "macs_per_frame": per_frame,
             "macs_by_part": {k: float(v) for k, v in macs.items()}, "input": list(s["shape"]),
             "macs_per_second_no_overlap": per_frame * s["fps"],
             "macs_per_second_with_overlap": per_frame * s["fps"] * passes[name]}
        if name == "V2N":
            r["macs_per_clip"] = per_frame * V2N_FPS * V2N_CLIP_S
        print(f"\n{name}: {params:,} parameters; {per_frame:.3e} MACs per frame; per second of video "
              f"{r['macs_per_second_with_overlap']:.3e} with its test-time overlap, "
              f"{r['macs_per_second_no_overlap']:.3e} without"
              + (f"; {r['macs_per_clip']:.3e} per 1-s clip" if name == "V2N" else ""))
        for label, dtype in precisions.items():
            try:
                sp = model_speed(model, s["shape"], args.device, dtype, args.reps)
            except RuntimeError as e:                 # e.g. out of memory
                sp = {"error": str(e).splitlines()[0]}
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            if "error" not in sp:
                gpu_s_per_frame = 1.0 / sp["frames_per_second"]
                sp["rtf_no_overlap"] = gpu_s_per_frame * s["fps"]
                sp["rtf_with_overlap"] = sp["rtf_no_overlap"] * passes[name]
                txt = (f"{sp['frames_per_second']:.0f} frames/s; model seconds per second of video: "
                       f"{sp['rtf_with_overlap']:.4f} (with overlap), {sp['rtf_no_overlap']:.4f} (no overlap)")
                if "peak_memory_gb" in sp:
                    txt += f"; peak memory {sp['peak_memory_gb']:.2f} GB"
            else:
                txt = sp["error"]
            r["speed_" + label] = sp
            print(f"  {label}: {txt}")
        res[name] = r
        model.cpu()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    k, v = res["KeyWise"], res["V2N"]
    print(f"\nV2N / KeyWise: parameters {v['parameters'] / k['parameters']:.1f}x; MACs per second of video"
          f" {v['macs_per_second_with_overlap'] / k['macs_per_second_with_overlap']:.1f}x (each with its"
          f" test-time overlap), {v['macs_per_second_no_overlap'] / k['macs_per_second_no_overlap']:.1f}x"
          f" (no overlap)")
    for label in precisions:
        a, b = k.get("speed_" + label, {}), v.get("speed_" + label, {})
        if "rtf_with_overlap" in a and "rtf_with_overlap" in b:
            print(f"  {label}: model time per second of video {b['rtf_with_overlap'] / a['rtf_with_overlap']:.1f}x"
                  f" (each with its test-time overlap), {b['rtf_no_overlap'] / a['rtf_no_overlap']:.1f}x (no overlap)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
