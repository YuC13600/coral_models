---
license: gpl-3.0
tags:
  - coral-reef
  - re-identification
  - metric-learning
  - dinov3
  - pytorch
datasets:
  - custom
pipeline_tag: image-feature-extraction
---

# Coral Re-ID: DINOv3 ViT-S+/16 (Efficient)

Fine-tuned DINOv3 ViT-S+/16 for underwater coral individual re-identification. This is the **most efficient model** in the project, achieving **81.1% N-Benchmark Top-1 accuracy** with only ~22M parameters and ~2h training time.

## Model Details

| | |
|---|---|
| **Architecture** | DINOv3 ViT-S+/16 (~22M params) |
| **Backbone loader** | timm (`vit_small_plus_patch16_dinov3`) |
| **Input size** | 512 x 512 |
| **Embedding dim** | 768 |
| **Backbone output dim** | 384 |
| **Head** | MLP (384 → 512 → 768, BatchNorm, Dropout 0.3) |

## Training Configuration

| | |
|---|---|
| **Loss** | Triplet Loss (margin=0.3) + Hard Mining |
| **Sampler** | MPerClassSampler (m=2) |
| **Batch size** | 16 (accumulation steps: 8, effective batch: 128) |
| **Optimizer** | AdamW (weight_decay=1e-4) |
| **Gradient clipping** | 1.0 |
| **Early stopping** | patience=6, delta=0.0005 |
| **Total epochs** | 63 |
| **Training time** | ~2.0 hours (single GPU) |

### Progressive Unfreezing (4-phase)

| Phase | Layers | LR | Max Epochs |
|-------|--------|----|------------|
| 1 — Head only | 0 (head only) | 3e-4 | 20 |
| 2 — Last 2 blocks | 2 | 5e-5 | 20 |
| 3 — Last 4 blocks | 4 | 1.5e-5 | 15 |
| 4 — Last 6 blocks | 6 | 1e-5 | 15 |

Phase 2 LR was reduced from the default 8e-5 to 5e-5 to prevent early stopping from triggering too soon, giving Phase 3 a better starting point. Phase 4 then further unlocks the model's capacity.

## Evaluation Results (N-Benchmark)

Cross-year matching: 2022 (reference) vs 2023 (query), areas 37-40.

| Area | Queries | Top-1 | Top-3 | Top-5 | Avg Rank |
|------|---------|-------|-------|-------|----------|
| 37 | 32 | 81.2% | 93.8% | 96.9% | 1.56 |
| 38 | 31 | 77.4% | 90.3% | 93.5% | 1.90 |
| 39 | 27 | 85.2% | 92.6% | 96.3% | 1.37 |
| 40 | 37 | 81.1% | 91.9% | 94.6% | 1.57 |
| **Overall** | **127** | **81.1%** | **92.1%** | **95.3%** | **1.61** |

- **Val loss**: 0.1604

## Comparison with Best Model

| Metric | Best (DINOv2 ViT-B) | This model | Difference |
|--------|---------------------|------------|------------|
| Top-1 | 86.6% | 81.1% | -5.5% |
| Parameters | ~86.6M | ~22M | **-75%** |
| Training time | ~7.2h | ~2.0h | **-72%** |
| Model file size | 339 MB | 112 MB | **-67%** |
| Inference tokens | 1369 (patch14) | 1024 (patch16) | -25% |

## Files

| File | Description |
|------|-------------|
| `best_model_20260306_233824.pt` | Best checkpoint (lowest val loss during training) |
| `final_model_20260306_233824.pt` | Final checkpoint (last epoch) |
| `dinov3_vitsplus_tune_02_p2lr5_4ph.yaml` | Full training config |

## Usage

```python
import torch
from coral_reid.config import ExperimentConfig
from coral_reid.models.coral_model import CoralReIDModel

config = ExperimentConfig.from_yaml("dinov3_vitsplus_tune_02_p2lr5_4ph.yaml")
model = CoralReIDModel.from_config(config.backbone, config.head)
model.load("best_model_20260306_233824.pt", map_location="cpu")
model.eval()

# Extract embedding
embedding = model(image_tensor)  # (1, 768)
```

Or with the standalone script (no `coral_reid` dependency):

```bash
uv run python extract_features.py \
    --model dinov3_vitsplus_efficient/best_model_20260306_233824.pt \
    --input /path/to/image.jpg
```

## Citation

Part of the coral re-identification research for long-term ecological monitoring at Xiaoliuqiu, Green Island, and Northeastern Taiwan.
