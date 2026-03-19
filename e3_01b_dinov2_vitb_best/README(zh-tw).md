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

# 珊瑚個體辨識：DINOv2 ViT-B/14（最佳精度）

針對水下珊瑚個體辨識微調的 DINOv2 ViT-B/14 模型。此為本專案中的**最強模型**，達到 **86.6% N-Benchmark Top-1 準確率**。

## 模型規格

| | |
|---|---|
| **架構** | DINOv2 ViT-B/14 (86.6M 參數) |
| **Backbone 載入方式** | timm (`vit_base_patch14_dinov2`) |
| **輸入尺寸** | 518 x 518 |
| **嵌入維度** | 1280 |
| **Backbone 輸出維度** | 768 |
| **Head** | MLP (768 → 1024 → 1280, BatchNorm, Dropout 0.3) |

## 訓練配置

| | |
|---|---|
| **損失函數** | Triplet Loss (margin=0.3) + Hard Mining |
| **取樣器** | AreaAwareSampler (area_ratio=0.75) |
| **批次大小** | 16（累積步數：8，等效批次：128） |
| **優化器** | AdamW (weight_decay=1e-4) |
| **梯度裁剪** | 1.0 |
| **Early stopping** | patience=6, delta=0.0005 |
| **總 epochs** | 56 |
| **訓練時間** | 約 7.2 小時（單 GPU） |

### 漸進式解凍（4 階段）

| 階段 | 解凍層數 | 學習率 | 最大 Epochs |
|------|----------|--------|-------------|
| 1 — 僅 Head | 0（僅 head） | 3e-4 | 20 |
| 2 — 最後 2 blocks | 2 | 8e-5 | 15 |
| 3 — 最後 4 blocks | 4 | 3e-5 | 12 |
| 4 — 最後 6 blocks | 6 | 1e-5 | 15 |

### AreaAwareSampler

每個訓練批次由 75% 同區域珊瑚與 25% 跨區域珊瑚組成。此設計對齊 N-Benchmark 的評估方式（區域內匹配），提供來自同一珊瑚礁區域中視覺上相似的更困難負樣本。

## 評估結果（N-Benchmark）

跨年匹配：2022（參考集）vs 2023（查詢集），區域 37-40。

| 區域 | 查詢數 | Top-1 | Top-3 | Top-5 | 平均排名 |
|------|--------|-------|-------|-------|----------|
| 37 | 32 | 93.8% | 96.9% | 96.9% | 1.28 |
| 38 | 31 | 80.6% | 100.0% | 100.0% | 1.19 |
| 39 | 27 | 85.2% | 92.6% | 96.3% | 1.44 |
| 40 | 37 | 86.5% | 97.3% | 97.3% | 1.30 |
| **整體** | **127** | **86.6%** | **96.9%** | **97.6%** | **1.30** |

- **最差排名**：9（所有正確匹配均在前 9 名內）
- **驗證損失**：0.1604

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `best_model_20260308_110634.pt` | 最佳 checkpoint（訓練期間最低驗證損失） |
| `final_model_20260308_110634.pt` | 最終 checkpoint（最後一個 epoch） |
| `e3_01b_same_area_neg_075.yaml` | 完整訓練配置 |

## 使用方式

```python
import torch
from coral_reid.config import ExperimentConfig
from coral_reid.models.coral_model import CoralReIDModel

config = ExperimentConfig.from_yaml("e3_01b_same_area_neg_075.yaml")
model = CoralReIDModel.from_config(config.backbone, config.head)
model.load("best_model_20260308_110634.pt", map_location="cpu")
model.eval()

# 提取嵌入向量
embedding = model(image_tensor)  # (1, 1280)
```

或使用獨立腳本（不需要 `coral_reid`）：

```bash
uv run python extract_features.py \
    --model e3_01b_dinov2_vitb_best/best_model_20260308_110634.pt \
    --input /path/to/image.jpg
```

## 引用

本模型為珊瑚個體辨識研究的一部分，用於小琉球、綠島及東北角珊瑚礁的長期生態監測。
