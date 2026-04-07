"""
utils/fft_utils.py — Frequency feature extraction for DeepTrace.

Changes vs original:
  - Per-channel FFT (R, G, B separately) instead of single-channel grayscale.
    Deepfake generators often leave channel-specific frequency residuals.
  - Log-magnitude normalised per-channel to [0, 255] → ToTensor gives [0, 1].
  - fftshift centres the DC component for a more interpretable, learnable map.
  - Optional: high-frequency emphasis (subtract low-freq mean) which amplifies
    the GAN-fingerprint band above 64 cycles/image.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def compute_fft_image(
    pil_img: Image.Image,
    size: int,
    emphasise_high_freq: bool = True,
) -> Image.Image:
    """
    Convert an RGB PIL image to a 3-channel log-FFT magnitude map.

    Parameters
    ----------
    pil_img : PIL.Image  Input image (any mode, converted to RGB).
    size    : int        Target spatial size (same as IMG_SIZE).
    emphasise_high_freq : bool
        If True, subtract the per-channel mean of the low-frequency region
        (centre 8×8 patch) before normalisation so the network focuses on
        the high-frequency GAN fingerprint band.

    Returns
    -------
    PIL.Image  RGB image of shape (size, size) representing log-FFT magnitude.
    """
    img = pil_img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)          # (H, W, 3)

    out_channels: list[np.ndarray] = []

    for c in range(3):
        channel = arr[:, :, c]

        # 2-D FFT + shift DC to centre
        fft    = np.fft.fft2(channel)
        fshift = np.fft.fftshift(fft)

        # Log magnitude (avoids log(0) with +1)
        magnitude = np.log1p(np.abs(fshift))       # (H, W)

        if emphasise_high_freq:
            # Suppress the low-frequency region (centre patch) so the
            # network learns from GAN-artifact bands, not the DC lump.
            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            r = 4  # radius of low-freq patch to neutralise
            lf_mean = magnitude[cy - r: cy + r, cx - r: cx + r].mean()
            magnitude = np.clip(magnitude - lf_mean * 0.5, 0, None)

        # Normalise to [0, 255]
        mn, mx = magnitude.min(), magnitude.max()
        norm = ((magnitude - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
        out_channels.append(norm)

    fft_rgb = np.stack(out_channels, axis=2)        # (H, W, 3)
    return Image.fromarray(fft_rgb, mode="RGB")