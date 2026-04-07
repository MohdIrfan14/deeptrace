"""Dataset transforms and DeepfakeDataset for DeepTrace.

This file provides the train/eval transforms and a Dataset that returns
both spatial and frequency-domain tensors together with the label.
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from config import cfg
from utils.fft_utils import compute_fft_image

# ImageNet stats used for both branches
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def get_train_spatial_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.10),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.20, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.40),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.08)),
    ])


def get_eval_spatial_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_freq_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def _collect_paths(data_dir: str) -> Tuple[List[str], List[int]]:
    paths: List[str] = []
    labels: List[int] = []
    ext_ok = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    for label, folder_name in enumerate(["real", "fake"]):
        folder = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Expected folder not found: {folder}")
        for fname in sorted(os.listdir(folder)):
            if os.path.splitext(fname)[1].lower() in ext_ok:
                paths.append(os.path.join(folder, fname))
                labels.append(label)

    return paths, labels


def _split(paths: List[str], labels: List[int], split: str, seed: int) -> Tuple[List[str], List[int]]:
    rng = random.Random(seed)
    real_idx = [i for i, l in enumerate(labels) if l == 0]
    fake_idx = [i for i, l in enumerate(labels) if l == 1]

    def _do_split(indices: List[int]):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = max(1, int(n * cfg.VAL_RATIO))
        n_test = max(1, int(n * cfg.TEST_RATIO))
        if split == "train":
            return shuffled[: n - n_val - n_test]
        elif split == "val":
            return shuffled[n - n_val - n_test: n - n_test]
        else:
            return shuffled[n - n_test:]

    chosen = _do_split(real_idx) + _do_split(fake_idx)
    return [paths[i] for i in chosen], [labels[i] for i in chosen]


class DeepfakeDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train") -> None:
        assert split in {"train", "val", "test"}, f"Unknown split: {split}"
        all_paths, all_labels = _collect_paths(data_dir)
        self.paths, self.labels = _split(all_paths, all_labels, split, cfg.SEED)
        self.is_train = (split == "train")

        self._spatial_tf = get_train_spatial_transform() if self.is_train else get_eval_spatial_transform()
        self._freq_tf = get_freq_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        label = self.labels[idx]

        pil_rgb = Image.open(img_path).convert("RGB")
        spatial_tensor = self._spatial_tf(pil_rgb)
        freq_pil = compute_fft_image(pil_rgb, size=cfg.IMG_SIZE)
        freq_tensor = self._freq_tf(freq_pil)

        return spatial_tensor, freq_tensor, torch.tensor(label, dtype=torch.float32)

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        labels_arr = np.array(self.labels)
        n_real = int((labels_arr == 0).sum())
        n_fake = int((labels_arr == 1).sum())

        w_real = 1.0 / n_real if n_real > 0 else 1.0
        w_fake = 1.0 / n_fake if n_fake > 0 else 1.0

        sample_weights = np.where(labels_arr == 0, w_real, w_fake).astype(np.float64)
        return WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )