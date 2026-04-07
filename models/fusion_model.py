"""
models/fusion_model.py — Hybrid Spatial-Frequency Deepfake Detector.

Architecture changes vs original:
  1. Both branches: EfficientNet-B4 (was ResNet50 + ResNet18 mismatch).
     EfficientNet-B4 gives ~+5 % over ResNet50 on image classification benchmarks
     and is significantly better at preserving fine-grained texture artifacts.
  2. Cross-attention fusion (was plain concatenation).
     The spatial branch attends over frequency features so the model learns
     *which* frequency artifacts matter for each spatial region.
  3. Projection heads with BN + GELU (more stable than bare Linear).
  4. Reduced dropout (0.3) to match the new config.
  5. Separate first-conv weight initialisation for the frequency branch
     so it isn't biased toward ImageNet colour statistics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

# Feature dimension of EfficientNet-B4 avgpool output
_EFFB4_DIM = 1792


# ─────────────────────────────────────────────────────────────────── helpers ──

class _Projection(nn.Module):
    """Linear → BN → GELU → Dropout projection block."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """
    Single-head cross-attention: spatial queries, frequency keys & values.

    Residual connection keeps the spatial representation intact while
    selectively incorporating frequency evidence.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)

    def forward(self, spatial: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        q = self.q(spatial)
        k = self.k(freq)
        v = self.v(freq)
        # Element-wise attention (batch of scalars, no sequence axis needed)
        attn = torch.sigmoid(q * k * self.scale)  # (B, dim)
        attended = attn * v
        out = self.out_proj(attended)
        return self.norm(spatial + out)            # residual


# ──────────────────────────────────────────────────────────────── main model ──

class FusionModel(nn.Module):
    """
    Dual-branch deepfake detector.

    Inputs
    ------
    spatial : (B, 3, H, W)  RGB image normalised with ImageNet stats
    freq    : (B, 3, H, W)  3-channel log-FFT magnitude (R/G/B channels)

    Outputs
    -------
    logit : (B,)   raw (un-sigmoided) score; positive → fake
    feat  : (B, 2*proj_dim)  fused embedding (for Grad-CAM / analysis)
    """

    def __init__(
        self,
        pretrained_backbones: bool = True,
        proj_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        weights = EfficientNet_B4_Weights.DEFAULT if pretrained_backbones else None

        # ── Spatial branch ──────────────────────────────────────────────────
        _spatial = efficientnet_b4(weights=weights)
        # Keep everything except the classifier head (last Linear)
        self.spatial_features = _spatial.features          # ConvBnActivation stack
        self.spatial_pool     = _spatial.avgpool           # AdaptiveAvgPool2d(1,1)

        # ── Frequency branch ────────────────────────────────────────────────
        _freq = efficientnet_b4(weights=weights)
        self.freq_features = _freq.features
        self.freq_pool      = _freq.avgpool

        # Re-initialise the very first conv of the frequency branch.
        # ImageNet pretrained weights assume RGB statistics; FFT magnitude maps
        # have a different distribution so random init is better here.
        nn.init.kaiming_normal_(
            self.freq_features[0][0].weight, mode="fan_out", nonlinearity="relu"
        )

        # ── Projection heads ────────────────────────────────────────────────
        self.spatial_proj = _Projection(_EFFB4_DIM, proj_dim, dropout)
        self.freq_proj    = _Projection(_EFFB4_DIM, proj_dim, dropout)

        # ── Cross-attention fusion ──────────────────────────────────────────
        self.fusion_attn = CrossAttentionFusion(proj_dim)

        # ── Classifier head ─────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    # ---------------------------------------------------------------- forward
    def _extract(self, features_block, pool, x: torch.Tensor) -> torch.Tensor:
        x = features_block(x)
        x = pool(x)
        return x.flatten(1)                        # (B, 1792)

    def forward(
        self, spatial: torch.Tensor, freq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s_raw = self._extract(self.spatial_features, self.spatial_pool, spatial)
        f_raw = self._extract(self.freq_features,    self.freq_pool,    freq)

        s_proj = self.spatial_proj(s_raw)          # (B, 512)
        f_proj = self.freq_proj(f_raw)             # (B, 512)

        # Spatial attends over frequency
        s_attended = self.fusion_attn(s_proj, f_proj)   # (B, 512)

        fused = torch.cat([s_attended, f_proj], dim=1)  # (B, 1024)
        logit = self.classifier(fused).squeeze(1)       # (B,)

        return logit, fused