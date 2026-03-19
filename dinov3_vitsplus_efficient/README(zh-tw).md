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

# 珊瑚個體辨識：DINOv3 ViT-S+/16（高效率）

針對水下珊瑚個體辨識微調的 DINOv3 ViT-S+/16 模型。此為本專案中的**最高效率模型**，僅以約 22M 參數和約 2 小時訓練時間，達到 **81.1% N-Benchmark Top-1 準確率**。

## 模型規格

| | |
|---|---|
| **架構** | DINOv3 ViT-S+/16 (~22M 參數) |
| **Backbone 載入方式** | timm (`vit_small_plus_patch16_dinov3`) |
| **輸入尺寸** | 512 x 512 |
| **嵌入維度** | 768 |
| **Backbone 輸出維度** | 384 |
| **Head** | MLP (384 → 512 → 768, BatchNorm, Dropout 0.3) |

## 訓練配置

| | |
|---|---|
| **損失函數** | Triplet Loss (margin=0.3) + Hard Mining |
| **取樣器** | MPerClassSampler (m=2) |
| **批次大小** | 16（累積步數：8，等效批次：128） |
| **優化器** | AdamW (weight_decay=1e-4) |
| **梯度裁剪** | 1.0 |
| **Early stopping** | patience=6, delta=0.0005 |
| **總 epochs** | 63 |
| **訓練時間** | 約 2.0 小時（單 GPU） |

### 漸進式解凍（4 階段）

| 階段 | 解凍層數 | 學習率 | 最大 Epochs |
|------|----------|--------|-------------|
| 1 — 僅 Head | 0（僅 head） | 3e-4 | 20 |
| 2 — 最後 2 blocks | 2 | 5e-5 | 20 |
| 3 — 最後 4 blocks | 4 | 1.5e-5 | 15 |
| 4 — 最後 6 blocks | 6 | 1e-5 | 15 |

Phase 2 學習率從預設的 8e-5 降至 5e-5，避免 early stopping 過早觸發，讓 Phase 3 有更好的起點。Phase 4 進一步釋放模型容量。

## 評估結果（N-Benchmark）

跨年匹配：2022（參考集）vs 2023（查詢集），區域 37-40。

| 區域 | 查詢數 | Top-1 | Top-3 | Top-5 | 平均排名 |
|------|--------|-------|-------|-------|----------|
| 37 | 32 | 81.2% | 93.8% | 96.9% | 1.56 |
| 38 | 31 | 77.4% | 90.3% | 93.5% | 1.90 |
| 39 | 27 | 85.2% | 92.6% | 96.3% | 1.37 |
| 40 | 37 | 81.1% | 91.9% | 94.6% | 1.57 |
| **整體** | **127** | **81.1%** | **92.1%** | **95.3%** | **1.61** |

- **驗證損失**：0.1604

## 與最強模型的比較

| 指標 | 最強模型 (DINOv2 ViT-B) | 本模型 | 差距 |
|------|------------------------|--------|------|
| Top-1 | 86.6% | 81.1% | -5.5% |
| 參數量 | ~86.6M | ~22M | **-75%** |
| 訓練時間 | ~7.2h | ~2.0h | **-72%** |
| 模型檔案大小 | 339 MB | 112 MB | **-67%** |
| 推論 tokens | 1369 (patch14) | 1024 (patch16) | -25% |

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `best_model_20260306_233824.pt` | 最佳 checkpoint（訓練期間最低驗證損失） |
| `final_model_20260306_233824.pt` | 最終 checkpoint（最後一個 epoch） |
| `dinov3_vitsplus_tune_02_p2lr5_4ph.yaml` | 完整訓練配置 |

## 使用方式

```python
import torch
from coral_reid.config import ExperimentConfig
from coral_reid.models.coral_model import CoralReIDModel

config = ExperimentConfig.from_yaml("dinov3_vitsplus_tune_02_p2lr5_4ph.yaml")
model = CoralReIDModel.from_config(config.backbone, config.head)
model.load("best_model_20260306_233824.pt", map_location="cpu")
model.eval()

# 提取嵌入向量
embedding = model(image_tensor)  # (1, 768)
```

或使用獨立腳本（不需要 `coral_reid`）：

```bash
uv run python extract_features.py \
    --model dinov3_vitsplus_efficient/best_model_20260306_233824.pt \
    --input /path/to/image.jpg
```

## 引用

本模型為珊瑚個體辨識研究的一部分，用於小琉球、綠島及東北角珊瑚礁的長期生態監測。
