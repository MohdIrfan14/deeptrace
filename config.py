"""
config.py — Central configuration for DeepTrace.

Key changes vs original:
  - IMG_SIZE: 128 → 224  (EfficientNet native size; preserves artifact detail)
  - BATCH_SIZE: 8 → 32   (stable gradient estimates)
  - NUM_EPOCHS: 10 → 50  (model needs time to converge)
  - DROPOUT: 0.5 → 0.3   (was over-regularising a model that was already under-fitting)
  - Both backbones now use EfficientNet-B4 (was ResNet50 + ResNet18 mismatch)
  - Added: WARMUP_EPOCHS, LABEL_SMOOTHING, GRAD_CLIP, MIXUP_ALPHA
"""

import os


class Config:
    # ------------------------------------------------------------------ paths
    PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR       = os.path.join(PROJECT_ROOT, "dataset")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    LOG_DIR        = os.path.join(PROJECT_ROOT, "logs")
    RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,        exist_ok=True)
    os.makedirs(RESULTS_DIR,    exist_ok=True)

    # ------------------------------------------------------------------ data
    IMG_SIZE    = 224   # was 128 — EfficientNet-B4 native; reveals subtle artifacts
    BATCH_SIZE  = 32    # was 8  — much stabler gradient signal
    NUM_WORKERS = 4

    TRAIN_RATIO = 0.70
    VAL_RATIO   = 0.15
    TEST_RATIO  = 0.15
    SEED        = 42

    # --------------------------------------------------------------- backbone
    # Both branches now use the same capacity backbone (was ResNet50 vs ResNet18)
    SPATIAL_BACKBONE  = "efficientnet_b4"
    FREQ_BACKBONE     = "efficientnet_b4"
    FUSION_HIDDEN_DIM = 512
    DROPOUT           = 0.3   # was 0.5 — too aggressive for under-fitting model

    # -------------------------------------------------------------- training
    NUM_EPOCHS               = 50    # was 10 — dual-branch model needs many more
    LEARNING_RATE            = 3e-4
    WEIGHT_DECAY             = 1e-4
    EARLY_STOPPING_PATIENCE  = 12   # monitor val AUC, not val loss
    WARMUP_EPOCHS            = 3    # linear LR warmup before cosine decay
    LABEL_SMOOTHING          = 0.05 # prevents overconfident logits
    GRAD_CLIP                = 1.0  # prevent exploding gradients
    MIXUP_ALPHA              = 0.2  # MixUp augmentation strength (0 = disabled)
    USE_AMP                  = True # mixed-precision training (2× speed on GPU)

    # pos_weight for BCEWithLogitsLoss — computed dynamically in train.py
    # from the actual class ratio so the model doesn't bias toward majority class.
    # Set a hard fallback here in case dataset is perfectly balanced.
    POS_WEIGHT_FALLBACK = 1.0

    # ----------------------------------------------------------- checkpoints
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_fusion_model.pth")


cfg = Config()