"""
Inference helpers for DeepTrace: single-file prediction reusing the same
preprocessing as val/test in utils/preprocessing.DeepfakeDataset (no augmentation).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import cfg
from models.fusion_model import FusionModel
from utils.fft_utils import compute_fft_image

# Loaded once by load_model() — used by predict_deepfake() for every request.
_device: Optional[torch.device] = None
_model: Optional[FusionModel] = None

# ImageNet stats used during train/test preprocessing.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# Same eval transform as DeepfakeDataset when split != "train"
_eval_spatial_transform = transforms.Compose(
    [
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)
_freq_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)

# Image extensions supported by training pipeline; video handled via frame extract.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}


def load_model() -> None:
    """
    Load FusionModel weights once at application startup (not per request).
    Mirrors test.py: FusionModel(pretrained_backbones=False) + state dict from cfg.BEST_MODEL_PATH.
    """
    global _device, _model

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = FusionModel(pretrained_backbones=False).to(_device)

    if not os.path.isfile(cfg.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Best model checkpoint not found at:\n  {cfg.BEST_MODEL_PATH}\n\n"
            "Create it by either:\n"
            "  1) Training: put images under dataset/real/ and dataset/fake/, then from the "
            "DeepTrace folder run: python train.py\n"
            "  2) Copying: place an existing best_fusion_model.pth into the checkpoints/ folder."
        )

    try:
        state = torch.load(cfg.BEST_MODEL_PATH, map_location=_device, weights_only=False)
    except TypeError:
        state = torch.load(cfg.BEST_MODEL_PATH, map_location=_device)
    _model.load_state_dict(state)
    _model.eval()


def _pil_from_video_frame(filepath: str) -> Image.Image:
    """Extract one representative frame (middle) as RGB PIL image."""
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "Video support requires opencv-python. Install with: pip install opencv-python"
        ) from e

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames > 0:
        mid = max(0, n_frames // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError("Could not read any frame from video.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _load_image_rgb(filepath: str) -> Image.Image:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in _VIDEO_EXTS:
        return _pil_from_video_frame(filepath)
    if ext not in _IMAGE_EXTS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Use images ({', '.join(sorted(_IMAGE_EXTS))}) or video ({', '.join(sorted(_VIDEO_EXTS))})."
        )
    img = Image.open(filepath).convert("RGB")
    return img


def preprocess_tensors(pil_rgb: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build spatial and frequency tensors matching DeepfakeDataset.__getitem__ (eval split).
    """
    spatial = _eval_spatial_transform(pil_rgb)
    freq_img = compute_fft_image(pil_rgb, size=cfg.IMG_SIZE)
    freq_tensor = _freq_transform(freq_img)
    return spatial, freq_tensor


def predict_deepfake(filepath: str, tta: bool = True) -> Tuple[str, float]:
    """
    Run binary deepfake prediction on an image file or video (middle frame).

    Prediction: sigmoid(logit) is P(fake). Label convention from training: 0=real, 1=fake.

    Args:
        filepath: Input image/video path.
        tta: Kept for API compatibility with app.py. Currently not used.

    Returns:
        result: "real" or "fake"
        confidence: float in [0, 1] for the predicted class (same as test.py threshold 0.5)
    """
    if _model is None or _device is None:
        raise RuntimeError("Model not loaded. Call load_model() before predict_deepfake().")

    if not filepath or not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    pil_rgb = _load_image_rgb(filepath)
    spatial, freq_tensor = preprocess_tensors(pil_rgb)

    spatial_batch = spatial.unsqueeze(0).to(_device)
    freq_batch = freq_tensor.unsqueeze(0).to(_device)

    # --- Prediction (same forward as test.py evaluate loop) ---
    with torch.no_grad():
        logits, _ = _model(spatial_batch, freq_batch)
        prob_fake = torch.sigmoid(logits).squeeze().item()

    prob_fake = float(np.clip(prob_fake, 0.0, 1.0))
    if prob_fake >= 0.5:
        result = "fake"
        confidence = prob_fake
    else:
        result = "real"
        confidence = 1.0 - prob_fake

    return result, confidence
