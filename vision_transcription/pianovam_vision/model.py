"""Visual Onsets-and-Frames network.

Per-frame 2D CNN encoder -> BiGRU over time -> onset / frame / (velocity) heads.
This is the acoustic Onsets-and-Frames idea with the spectrogram replaced by a
rectified keyboard image. The horizontal axis of the strip carries pitch, so the
encoder keeps some horizontal resolution before the temporal model.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from . import N_KEYS


def conv_block(cin: int, cout: int) -> nn.Sequential:
    # Downsample on the FIRST conv so the (memory-heavy) second conv runs at the
    # reduced resolution. This keeps activation memory small for big B*T batches.
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),  # downsample
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class FrameEncoder(nn.Module):
    """Encodes one warped strip into a feature vector, keeping horizontal info."""

    def __init__(self, in_ch: int, channels: List[int], feature_dim: int,
                 keep_width: int = 16):
        super().__init__()
        blocks = []
        c = in_ch
        for ch in channels:
            blocks.append(conv_block(c, ch))
            c = ch
        self.conv = nn.Sequential(*blocks)
        # Collapse height to 1, keep `keep_width` horizontal bins (pitch position).
        self.pool = nn.AdaptiveAvgPool2d((1, keep_width))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * keep_width, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,C,H,W) -> (N,D)
        x = self.conv(x)
        x = self.pool(x)
        return self.fc(x)


class VisualOnsetsFrames(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        m = cfg["model"]
        in_ch = 1 if cfg["keyboard"]["grayscale"] else 3
        self.use_velocity = bool(m["use_velocity"])

        # strip_bins: horizontal bins kept before the projection to one global
        # vector (16 = the paper's global variant; 88 = one bin per key, the
        # review ablation). Older checkpoints have no key and load as 16.
        self.encoder = FrameEncoder(
            in_ch, m["encoder_channels"], m["feature_dim"],
            keep_width=int(m.get("strip_bins", 16)),
        )
        self.gru = nn.GRU(
            input_size=m["feature_dim"],
            hidden_size=m["gru_hidden"],
            num_layers=m["gru_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=m["dropout"] if m["gru_layers"] > 1 else 0.0,
        )
        h = 2 * m["gru_hidden"]
        self.drop = nn.Dropout(m["dropout"])
        self.onset_head = nn.Linear(h, N_KEYS)
        # Frame head is conditioned on the (detached) onset prediction, as in
        # Onsets-and-Frames, which sharpens note starts.
        self.frame_head = nn.Linear(h + N_KEYS, N_KEYS)
        self.velocity_head = nn.Linear(h, N_KEYS) if self.use_velocity else None

    def forward(self, frames: torch.Tensor,
                skel: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        # skel is accepted for a uniform call signature; the global model has no
        # hand branch (hands are a per-key read-out, see TiledVisualOnsetsFrames).
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        feat = self.encoder(x).reshape(B, T, -1)
        g, _ = self.gru(feat)
        g = self.drop(g)

        onset_logits = self.onset_head(g)
        frame_in = torch.cat([g, torch.sigmoid(onset_logits).detach()], dim=-1)
        frame_logits = self.frame_head(frame_in)

        out = {"onset_logits": onset_logits, "frame_logits": frame_logits}
        if self.velocity_head is not None:
            out["velocity"] = torch.sigmoid(self.velocity_head(g))
        return out


class TiledEncoder(nn.Module):
    """Encodes a warped strip into ONE feature vector per key (ROI tile).

    The conv stack keeps horizontal resolution; we then pool the width into
    ``num_tiles`` bins (one ROI per piano key) and project each bin to a per-key
    embedding. This gives the temporal model an explicit, localised view of
    every key instead of a single global descriptor.
    """

    def __init__(self, in_ch: int, channels: List[int], tile_dim: int,
                 num_tiles: int):
        super().__init__()
        blocks = []
        c = in_ch
        for ch in channels:
            blocks.append(conv_block(c, ch))
            c = ch
        self.conv = nn.Sequential(*blocks)
        self.num_tiles = num_tiles
        # Collapse height to 1, keep exactly one horizontal bin per key (ROI).
        self.pool = nn.AdaptiveAvgPool2d((1, num_tiles))
        # 1x1 conv = a shared linear applied independently to each key's bin.
        self.proj = nn.Conv1d(c, tile_dim, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,C,H,W)->(N,K,Dt)
        x = self.conv(x)
        x = self.pool(x).squeeze(2)        # (N, C, K)
        x = self.act(self.proj(x))         # (N, Dt, K)
        return x.transpose(1, 2)           # (N, K, Dt)


class SkeletonKeyFeatures(nn.Module):
    """Hand landmarks -> one feature vector per key (no learned weights).

    Input ``skel`` (B, T, 2, 6, 4): for the left and right hand, the wrist and
    the five fingertips as [x, y, z, present] in the strip's normalised
    coordinates (see ``hands.py``). For every key and every landmark it measures
    where the landmark is relative to THAT key, with the real key shapes:
    signed horizontal offset from the key centre, a proximity bump, the distance
    to the key's surface, whether the landmark is on the key, how far along the
    keys it is, MediaPipe's depth, and whether the hand was detected. Because
    the strip is rectified, this is the same coordinate frame the per-key ROI
    tiles are cut from: landmark and tile meet exactly at the key.
    """

    N_FEATS = 7

    def __init__(self, n_landmarks: int = 6):
        super().__init__()
        from . import hands

        cx, hw, ln = hands.key_layout()
        self.register_buffer("cx", torch.tensor(cx, dtype=torch.float32), persistent=False)
        self.register_buffer("hw", torch.tensor(hw, dtype=torch.float32), persistent=False)
        self.register_buffer("ln", torch.tensor(ln, dtype=torch.float32), persistent=False)
        self.n_white = float(hands.N_WHITE)
        self.aspect = float(hands.KEY_ASPECT)
        self.out_dim = 2 * n_landmarks * self.N_FEATS

    def forward(self, skel: torch.Tensor) -> torch.Tensor:   # -> (B, T, K, out_dim)
        skel = torch.nan_to_num(skel.float())
        B, T = skel.shape[:2]
        x, y, z, pres = (skel[..., i:i + 1] for i in range(4))        # (B,T,2,L,1)
        dx = (x - self.cx) * self.n_white                              # white-key units
        out_x = (dx.abs() - self.hw * self.n_white).clamp(min=0)
        out_y = ((y - self.ln).clamp(min=0) + (-y).clamp(min=0)) * self.aspect
        dist = torch.sqrt(out_x ** 2 + out_y ** 2 + 1e-6)
        f = torch.stack([
            dx.clamp(-6, 6) / 6, torch.exp(-0.5 * dx ** 2), torch.exp(-dist),
            (dist < 0.01).float(), y.clamp(-0.5, 2.0).expand_as(dx),
            (10 * z).clamp(-1, 1).expand_as(dx), torch.ones_like(dx),
        ], dim=-1) * pres.unsqueeze(-1)                                # (B,T,2,L,K,F)
        return f.permute(0, 1, 4, 2, 3, 5).reshape(B, T, f.shape[4], -1)


class HandBranch(nn.Module):
    """Which hand and finger is on each key: 10 classes per key and frame
    (L1..L5, R1..R5), read out by ``hands.decide`` at a decoded note's frames.

    Input per key: the key's temporal feature from the shared BiGRU (what the
    key ROI saw -- the same feature that detects the onset) and/or the skeleton
    features of that key. A key ROI alone is narrow (the CNN sees ~3 white
    keys), while telling hands and fingers apart needs the whole hand, so the
    branch adds context ACROSS keys: residual dilated convolutions along the
    key axis (kernel 5, dilations 1-2-4-8: +/-30 keys, about a hand span each
    side) and one along time. A learned per-key embedding gives the pitch prior
    (low keys are mostly left hand). Weight-shared over keys like the rest of
    the model, so it stays small (~0.3 M parameters).
    """

    def __init__(self, video_dim: int, h: Dict[str, Any]):
        super().__init__()
        from . import N_KEYS as K, hands

        self.use_video = bool(h.get("video", True))
        self.skel = SkeletonKeyFeatures() if bool(h.get("skeleton", True)) else None
        d_in = (video_dim if self.use_video else 0) + (self.skel.out_dim if self.skel else 0)
        if d_in == 0:
            raise ValueError("model.hands needs video: true and/or skeleton: true")
        dim = int(h.get("dim", 96))
        k = int(h.get("key_kernel", 5))
        self.inp = nn.Linear(d_in, dim)
        self.key_embed = nn.Parameter(torch.randn(K, dim) * 0.02)
        self.key_convs = nn.ModuleList(
            nn.Conv1d(dim, dim, k, dilation=int(d), padding=int(d) * (k - 1) // 2)
            for d in h.get("key_dilations", [1, 2, 4, 8]))
        tk = int(h.get("time_kernel", 5))
        self.time_conv = nn.Conv1d(dim, dim, tk, padding=tk // 2) if tk > 1 else None
        self.drop = nn.Dropout(float(h.get("dropout", 0.1)))
        self.out = nn.Linear(dim, hands.N_CLASSES)

    def forward(self, g: torch.Tensor, skel: torch.Tensor | None) -> torch.Tensor:
        """g (B, T, K, Dg), skel (B, T, 2, 6, 4) or None -> logits (B, T, K, 10)."""
        return self.out(self.drop(self.encode(g, skel)))

    def encode(self, g: torch.Tensor, skel: torch.Tensor | None) -> torch.Tensor:
        """The hidden feature per key and frame (B, T, K, dim) that ``forward``
        classifies; the note-level model (hand_notes.py) reads it too. ``g`` may
        have 0 channels (no video: pass ``g.new_zeros(B, T, K, 0)``)."""
        B, T, K, _ = g.shape
        parts = [g] if self.use_video else []
        if self.skel is not None:
            if skel is None:               # no landmarks: every hand "not detected"
                skel = g.new_zeros(B, T, 2, 6, 4)
            parts.append(self.skel(skel).to(g.dtype))
        x = torch.relu(self.inp(torch.cat(parts, dim=-1)) + self.key_embed)   # (B,T,K,D)
        D = x.shape[-1]
        y = x.reshape(B * T, K, D).transpose(1, 2)                            # (BT,D,K)
        for conv in self.key_convs:
            y = y + torch.relu(conv(self.drop(y)))
        y = y.transpose(1, 2).reshape(B, T, K, D)
        if self.time_conv is not None:
            z = y.permute(0, 2, 3, 1).reshape(B * K, D, T)
            z = z + torch.relu(self.time_conv(self.drop(z)))
            y = z.reshape(B, K, D, T).permute(0, 3, 1, 2)
        return y


class TiledVisualOnsetsFrames(nn.Module):
    """Per-key ROI variant: each of the 88 keys gets its own feature + temporal
    stream (weight-shared across keys) and predicts its own onset/frame."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        m = cfg["model"]
        in_ch = 1 if cfg["keyboard"]["grayscale"] else 3
        self.use_velocity = bool(m["use_velocity"])
        self.num_tiles = N_KEYS                      # one ROI per key
        tile_dim = int(m.get("tile_feature_dim", 64))
        hidden = int(m.get("tile_gru_hidden", m["gru_hidden"]))

        self.encoder = TiledEncoder(
            in_ch, m["encoder_channels"], tile_dim, self.num_tiles
        )
        self.gru = nn.GRU(
            input_size=tile_dim,
            hidden_size=hidden,
            num_layers=m["gru_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=m["dropout"] if m["gru_layers"] > 1 else 0.0,
        )
        h = 2 * hidden
        self.drop = nn.Dropout(m["dropout"])
        # Heads act on a single key's feature -> one logit per key.
        self.onset_head = nn.Linear(h, 1)
        self.frame_head = nn.Linear(h + 1, 1)        # conditioned on onset
        self.velocity_head = nn.Linear(h, 1) if self.use_velocity else None
        # Hand/finger branch (KeyWise-hands, PianoVAM only); absent by default, so
        # checkpoints and outputs of the transcription models are unchanged.
        hcfg = m.get("hands") or {}
        self.hand_branch = HandBranch(h, hcfg) if hcfg.get("enable", False) else None

    def transcription_modules(self) -> List[nn.Module]:
        """Everything except the hand branch (frozen by train.freeze_transcription)."""
        return [mod for name, mod in self.named_children() if name != "hand_branch"]

    def forward(self, frames: torch.Tensor,
                skel: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = frames.shape
        K = self.num_tiles
        x = frames.reshape(B * T, C, H, W)
        feat = self.encoder(x).reshape(B, T, K, -1)  # (B,T,K,Dt)
        # Temporal model per key (shared weights): batch keys with the samples.
        feat = feat.permute(0, 2, 1, 3).reshape(B * K, T, -1)  # (B*K,T,Dt)
        g, _ = self.gru(feat)                        # (B*K,T,2H)
        g = self.drop(g)

        onset = self.onset_head(g)                   # (B*K,T,1)
        frame_in = torch.cat([g, torch.sigmoid(onset).detach()], dim=-1)
        frame = self.frame_head(frame_in)            # (B*K,T,1)

        onset_logits = onset.reshape(B, K, T).permute(0, 2, 1)  # (B,T,K)
        frame_logits = frame.reshape(B, K, T).permute(0, 2, 1)
        out = {"onset_logits": onset_logits, "frame_logits": frame_logits}
        if self.velocity_head is not None:
            vel = torch.sigmoid(self.velocity_head(g))
            out["velocity"] = vel.reshape(B, K, T).permute(0, 2, 1)
        if self.hand_branch is not None:
            gk = g.reshape(B, K, T, -1).permute(0, 2, 1, 3)          # (B,T,K,2H)
            out["hand_logits"] = self.hand_branch(gk, skel)          # (B,T,K,10)
        return out


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    arch = cfg["model"].get("arch", "strip")
    if arch == "tiled":
        return TiledVisualOnsetsFrames(cfg)
    if arch == "strip":
        if (cfg["model"].get("hands") or {}).get("enable", False):
            raise ValueError("model.hands needs model.arch: tiled (a per-key read-out)")
        return VisualOnsetsFrames(cfg)
    raise ValueError(f"Unknown model.arch={arch!r} (expected 'strip' or 'tiled')")
