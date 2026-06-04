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
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, stride=2, padding=1, bias=False),  # downsample
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


def build_model(cfg: Dict[str, Any]) -> VisualOnsetsFrames:
    return VisualOnsetsFrames(cfg)
