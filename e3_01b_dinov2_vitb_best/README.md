---
license: gpl-3.0
tags:
  - coral-reef
  - re-identification
  - metric-learning
  - dinov2
  - pytorch
datasets:
  - custom
pipeline_tag: image-feature-extraction
---

# Coral Re-ID: DINOv2 ViT-B/14 (Best Accuracy)

Fine-tuned DINOv2 ViT-B/14 for underwater coral individual re-identification. This is the **strongest model** in the project, achieving **86.6% N-Benchmark Top-1 accuracy**.

## Model Details

| | |
|---|---|
| **Architecture** | DINOv2 ViT-B/14 (86.6M params) |
| **Backbone loader** | timm (`vit_base_patch14_dinov2`) |
| **Input size** | 518 x 518 |
| **Embedding dim** | 1280 |
| **Backbone output dim** | 768 |
| **Head** | MLP (768 → 1024 → 1280, BatchNorm, Dropout 0.3) |

## Training Configuration

| | |
|---|---|
| **Loss** | Triplet Loss (margin=0.3) + Hard Mining |
| **Sampler** | AreaAwareSampler (area_ratio=0.75) |
| **Batch size** | 16 (accumulation steps: 8, effective batch: 128) |
| **Optimizer** | AdamW (weight_decay=1e-4) |
| **Gradient clipping** | 1.0 |
| **Early stopping** | patience=6, delta=0.0005 |
| **Total epochs** | 56 |
| **Training time** | ~7.2 hours (single GPU) |

### Progressive Unfreezing (4-phase)

| Phase | Layers | LR | Max Epochs |
|-------|--------|----|------------|
| 1 — Head only | 0 (head only) | 3e-4 | 20 |
| 2 — Last 2 blocks | 2 | 8e-5 | 15 |
| 3 — Last 4 blocks | 4 | 3e-5 | 12 |
| 4 — Last 6 blocks | 6 | 1e-5 | 15 |

### AreaAwareSampler

Each training batch is composed of 75% same-area corals and 25% cross-area corals. This aligns training distribution with the N-Benchmark evaluation protocol (within-area matching), providing harder negative examples from visually similar corals in the same reef area.

## Evaluation Results (N-Benchmark)

Cross-year matching: 2022 (reference) vs 2023 (query), areas 37-40.

| Area | Queries | Top-1 | Top-3 | Top-5 | Avg Rank |
|------|---------|-------|-------|-------|----------|
| 37 | 32 | 93.8% | 96.9% | 96.9% | 1.28 |
| 38 | 31 | 80.6% | 100.0% | 100.0% | 1.19 |
| 39 | 27 | 85.2% | 92.6% | 96.3% | 1.44 |
| 40 | 37 | 86.5% | 97.3% | 97.3% | 1.30 |
| **Overall** | **127** | **86.6%** | **96.9%** | **97.6%** | **1.30** |

- **Worst rank**: 9 (all correct matches within top 9)
- **Val loss**: 0.1604

## Files

| File | Description |
|------|-------------|
| `best_model_20260308_110634.pt` | Best checkpoint (lowest val loss during training) |
| `final_model_20260308_110634.pt` | Final checkpoint (last epoch) |
| `e3_01b_same_area_neg_075.yaml` | Full training config |

## Usage

```python
import torch
from coral_reid.config import ExperimentConfig
from coral_reid.models.coral_model import CoralReIDModel

config = ExperimentConfig.from_yaml("e3_01b_same_area_neg_075.yaml")
model = CoralReIDModel.from_config(config.backbone, config.head)
model.load("best_model_20260308_110634.pt", map_location="cpu")
model.eval()

# Extract embedding
embedding = model(image_tensor)  # (1, 1280)
```

Or with the standalone script (no `coral_reid` dependency):

```bash
uv run python extract_features.py \
    --model e3_01b_dinov2_vitb_best/best_model_20260308_110634.pt \
    --input /path/to/image.jpg
```

## Citation

Part of the coral re-identification research for long-term ecological monitoring at Xiaoliuqiu, Green Island, and Northeastern Taiwan.
