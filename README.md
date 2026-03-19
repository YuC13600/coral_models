# Coral Re-Identification Models

Fine-tuned models for underwater coral individual re-identification across multiple years.

This repository contains standalone inference scripts and benchmark data. Model weights are hosted on [Hugging Face](https://huggingface.co/YuC13600/coral_models).

> **Model weights**: Download from [Hugging Face](https://huggingface.co/YuC13600/coral_models) and place `.pt` files into the corresponding model directories.

## Best Models

### Best Accuracy — E3-01b DINOv2 ViT-B/14

| | |
|---|---|
| **N-Benchmark Top-1** | **86.6%** (110/127) |
| Top-3 / Top-5 / Top-10 | 96.9% / 97.6% / 100.0% |
| Avg Rank / Worst Rank | 1.30 / 9 |
| Backbone | DINOv2 ViT-B/14 (86.6M params, timm 518×518) |
| Loss | Triplet (margin=0.3) + Hard Mining |
| Sampler | AreaAwareSampler (area_ratio=0.75) |
| Training | 4-phase progressive unfreezing, 56 epochs, ~7.2h |
| Embedding | 1280-d, L2-normalized |
| Files | `e3_01b_dinov2_vitb_best/` |

### Most Efficient — DINOv3 ViT-S+/16

| | |
|---|---|
| **N-Benchmark Top-1** | **81.1%** (103/127) |
| Top-3 / Top-5 / Top-10 | 92.1% / 95.3% / 99.2% |
| Avg Rank | 1.61 |
| Backbone | DINOv3 ViT-S+/16 (~22M params, timm 512×512) |
| Loss | Triplet (margin=0.3) + Hard Mining |
| Sampler | MPerClassSampler (m=2) |
| Training | 4-phase progressive unfreezing, 63 epochs, ~2.0h |
| Embedding | 768-d, L2-normalized |
| Files | `dinov3_vitsplus_efficient/` |

### Comparison

| Metric | Best Accuracy | Most Efficient | Difference |
|--------|--------------|----------------|------------|
| Top-1 | 86.6% | 81.1% | -5.5% |
| Parameters | ~86.6M | ~22M | **-75%** |
| Model size | 339 MB | 112 MB | **-67%** |
| Training time | ~7.2h | ~2.0h | **-72%** |
| Inference tokens | 1369 (patch14) | 1024 (patch16) | -25% |

## Quick Start

```bash
# Download model weights from Hugging Face
# Place .pt files into e3_01b_dinov2_vitb_best/ and dinov3_vitsplus_efficient/

# Install dependencies (standalone, no coral_reid needed)
uv sync

# Extract features from a single image
uv run python extract_features.py \
    --model e3_01b_dinov2_vitb_best/best_model_20260308_110634.pt \
    --input /path/to/image.jpg

# Extract features from a directory
uv run python extract_features.py \
    --model e3_01b_dinov2_vitb_best/best_model_20260308_110634.pt \
    --input /path/to/images/ \
    --output features.h5

```

## N-Benchmark Per-Area Results

### E3-01b DINOv2 ViT-B/14 (Best)

| Area | Queries | Top-1 | Top-3 | Top-5 | Avg Rank |
|------|---------|-------|-------|-------|----------|
| 37 | 32 | 93.8% | 96.9% | 96.9% | 1.28 |
| 38 | 31 | 80.6% | 100.0% | 100.0% | 1.19 |
| 39 | 27 | 85.2% | 92.6% | 96.3% | 1.44 |
| 40 | 37 | 86.5% | 97.3% | 97.3% | 1.30 |
| **Overall** | **127** | **86.6%** | **96.9%** | **97.6%** | **1.30** |

### DINOv3 ViT-S+/16 (Efficient)

| Area | Queries | Top-1 | Top-3 | Top-5 | Avg Rank |
|------|---------|-------|-------|-------|----------|
| 37 | 32 | 81.2% | 93.8% | 96.9% | 1.56 |
| 38 | 31 | 77.4% | 90.3% | 93.5% | 1.90 |
| 39 | 27 | 85.2% | 92.6% | 96.3% | 1.37 |
| 40 | 37 | 81.1% | 91.9% | 94.6% | 1.57 |
| **Overall** | **127** | **81.1%** | **92.1%** | **95.3%** | **1.61** |

## Full Model History

### Model Comparison Table

| Model Name | Arch | Backbone | Loss | Mining | Same Area Neg | Image | Test Acc | Test Loss | Val Loss | N-Bench Avg | A37 | A38 | A39 | A40 | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pre-trained | - | DINOv2-B/14 | - | - | - | bbox | - | - | - | 29.48% | 28.12% | 35.48% | 29.63% | 24.32% | - |
| 20250812_152526 | old | DINOv2-B/14 | Triplet | pre-composed | ❌ | bbox | 92.6% | 0.1659 | - | 48.25% | 50.00% | 51.61% | 48.15% | 43.24% | ~16h |
| 20251007_133126 | old | DINOv2-B/14 | Triplet | pre-composed | ✅ | bbox | 88.8% | 0.2523 | - | 39.32% | 46.88% | 41.94% | 33.33% | 35.14% | ~16h |
| 20251008_094017 | old | DINOv2-B/14 | Triplet | pre-composed | ✅ | bbox | 90.4% | 0.1636 | - | 40.19% | 37.50% | 48.39% | 37.04% | 37.84% | ~16h |
| 20251014_183603 | old | DINOv2-B/14 | Triplet | pre-composed | ❌ | bbox | 92.8% | 0.1012 | - | 40.97% | 37.50% | 38.71% | 44.44% | 43.24% | ~16h |
| Pre-trained | - | DINOv2-B/14 | - | - | - | whole | - | - | - | 50.88% | 34.38% | 54.84% | 62.96% | 51.35% | - |
| 20251015_165008 | old | DINOv2-B/14 | Triplet | pre-composed | ✅ | whole | 92.7% | 0.1330 | 0.1006 | 64.43% | 62.50% | 61.29% | 55.56% | 78.38% | ~16h |
| 20251016_133229 | old | DINOv2-B/14 | Triplet | pre-composed | ❌ | whole | 97.9% | 0.0429 | - | 63.31% | 56.25% | 58.06% | 74.07% | 64.86% | ~16h |
| **20260308_110634** | **new** | **DINOv2-B/14 (timm 518)** | **Triplet** | **dynamic (PML)** | **AreaAware 0.75** | whole | - | - | **0.1604** | **86.6%** | **93.8%** | **80.6%** | **85.2%** | **86.5%** | **~7.2h** |
| **20260306_233824** | **new** | **DINOv3-S+/16 (timm 512)** | **Triplet** | **dynamic (PML)** | ❌ | whole | - | - | **0.1604** | **81.1%** | **81.2%** | **77.4%** | **85.2%** | **81.1%** | **~2.0h** |

### Column Descriptions

| Column | Description |
| --- | --- |
| Arch | `old` = old_repo implementation, `new` = refactored modular architecture |
| Backbone | Feature extractor (DINOv2-B/14, DINOv3-S+/16, etc.) |
| Loss | Loss function (Triplet, ArcFace, CosFace, Circle, Contrastive, etc.) |
| Mining | Sample mining: `pre-composed` = fixed triplets, `dynamic (PML)` = MPerClassSampler |
| Same Area Neg | Whether negatives restricted to same geographic area (`AreaAware 0.75` = 75% same area) |
| Image | `bbox` = EXIF bounding box crop, `whole` = full image |
| Test Acc | Test set accuracy (old arch only, measures pos_dist < neg_dist) |
| Val Loss | Best validation loss during training |
| N-Bench Avg | N-Benchmark Top-1 accuracy averaged across areas 37-40 |

### Architecture Differences

| Feature | Old Architecture | New Architecture |
| --- | --- | --- |
| Dataset Output | `(anchor, pos, neg)` - 3 images | `(image, label)` - 1 image |
| Triplet Formation | Pre-composed before training | Dynamic mining per batch |
| Batch Sampler | Random | MPerClassSampler (m=2) |
| Loss Function | Custom TripletLossWithMining | PML TripletMarginLoss |
| Samples per Epoch | ~50,000 triplets x 3 images | ~4,000 images |
| Training Speed | ~23 min/epoch | ~1.5 min/epoch |
| Same Area Negatives | Implemented | Implemented (AreaAwareSampler) |

> **N-Benchmark (Nearest Benchmark)**: Top-1 accuracy rate of identifying the correct coral when comparing specimens in areas 37-40 across 2022 and 2023.

## Project Structure

```
coral_models/
├── pyproject.toml                        # uv environment (standalone)
├── extract_features.py                   # Feature extraction script
├── e3_01b_dinov2_vitb_best/              # Best accuracy model (86.6%)
│   ├── e3_01b_same_area_neg_075.yaml     #   Training config
│   ├── README.md
│   └── README(zh-tw).md
├── dinov3_vitsplus_efficient/            # Most efficient model (81.1%)
│   ├── dinov3_vitsplus_tune_02_p2lr5_4ph.yaml  #   Training config
│   ├── README.md
│   └── README(zh-tw).md
├── 2022sample/                          # N-Benchmark reference images
└── 2023sample/                          # N-Benchmark query images
```

> Model weights (`.pt` files) and legacy models are hosted on [Hugging Face](https://huggingface.co/YuC13600/coral_models).

## License

This project is licensed under GPL-3.0.

Based on DINOv2 and DINOv3 by Meta Platforms, Inc. (Apache License 2.0).
