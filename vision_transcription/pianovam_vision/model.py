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

        self.encoder = FrameEncoder(
            in_ch, m["encoder_channels"], m["feature_dim"]
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

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
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

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        return out


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    arch = cfg["model"].get("arch", "strip")
    if arch == "tiled":
        return TiledVisualOnsetsFrames(cfg)
    if arch == "strip":
        return VisualOnsetsFrames(cfg)
    raise ValueError(f"Unknown model.arch={arch!r} (expected 'strip' or 'tiled')")
