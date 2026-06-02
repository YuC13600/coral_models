# 珊瑚個體辨識模型

針對水下珊瑚個體跨年辨識的微調模型。

本專案包含獨立推論腳本與基準測試資料，模型權重託管於 [Hugging Face](https://huggingface.co/YuC13600/coral_models)。

> **模型權重**：請從 [Hugging Face](https://huggingface.co/YuC13600/coral_models) 下載 `.pt` 檔案，並放入對應的模型目錄中。

## 最佳模型

### 最高精度 — E3-01b DINOv2 ViT-B/14

| | |
|---|---|
| **N-Benchmark Top-1** | **86.6%** (110/127) |
| Top-3 / Top-5 / Top-10 | 96.9% / 97.6% / 100.0% |
| 平均排名 / 最差排名 | 1.30 / 9 |
| Backbone | DINOv2 ViT-B/14 (86.6M 參數, timm 518×518) |
| 損失函數 | Triplet (margin=0.3) + Hard Mining |
| 取樣器 | AreaAwareSampler (area_ratio=0.75) |
| 訓練 | 4 階段漸進式解凍，56 epochs，約 7.2 小時 |
| 嵌入維度 | 1280-d，L2 正規化 |
| 檔案 | `e3_01b_dinov2_vitb_best/` |

### 最高效率 — DINOv3 ViT-S+/16

| | |
|---|---|
| **N-Benchmark Top-1** | **81.1%** (103/127) |
| Top-3 / Top-5 / Top-10 | 92.1% / 95.3% / 99.2% |
| 平均排名 | 1.61 |
| Backbone | DINOv3 ViT-S+/16 (~22M 參數, timm 512×512) |
| 損失函數 | Triplet (margin=0.3) + Hard Mining |
| 取樣器 | MPerClassSampler (m=2) |
| 訓練 | 4 階段漸進式解凍，63 epochs，約 2.0 小時 |
| 嵌入維度 | 768-d，L2 正規化 |
| 檔案 | `dinov3_vitsplus_efficient/` |

### 模型比較

| 指標 | 最高精度 | 最高效率 | 差距 |
|------|---------|---------|------|
| Top-1 | 86.6% | 81.1% | -5.5% |
| 參數量 | ~86.6M | ~22M | **-75%** |
| 模型大小 (FP32) | 339 MB | 112 MB | **-67%** |
| 模型大小 (FP16) | 170 MB | 56 MB | **-67%** |
| 訓練時間 | ~7.2h | ~2.0h | **-72%** |
| 推論 tokens | 1369 (patch14) | 1024 (patch16) | -25% |

## 快速開始

```bash
# 從 Hugging Face 下載模型權重
# 將 .pt 檔案放入 e3_01b_dinov2_vitb_best/ 和 dinov3_vitsplus_efficient/

# 安裝依賴（獨立環境，不需要 coral_reid）
uv sync

# 提取單張圖片特徵
uv run python extract_features.py \
    --model e3_01b_dinov2_vitb_best/best_model_20260308_110634.pt \
    --input /path/to/image.jpg

# 提取整個目錄的特徵
uv run python extract_features.py \
    --model e3_01b_dinov2_vitb_best/best_model_20260308_110634.pt \
    --input /path/to/images/ \
    --output features.h5

```

## N-Benchmark 各區域結果

### E3-01b DINOv2 ViT-B/14（最佳）

| 區域 | 查詢數 | Top-1 | Top-3 | Top-5 | 平均排名 |
|------|--------|-------|-------|-------|----------|
| 37 | 32 | 93.8% | 96.9% | 96.9% | 1.28 |
| 38 | 31 | 80.6% | 100.0% | 100.0% | 1.19 |
| 39 | 27 | 85.2% | 92.6% | 96.3% | 1.44 |
| 40 | 37 | 86.5% | 97.3% | 97.3% | 1.30 |
| **整體** | **127** | **86.6%** | **96.9%** | **97.6%** | **1.30** |

### DINOv3 ViT-S+/16（高效率）

| 區域 | 查詢數 | Top-1 | Top-3 | Top-5 | 平均排名 |
|------|--------|-------|-------|-------|----------|
| 37 | 32 | 81.2% | 93.8% | 96.9% | 1.56 |
| 38 | 31 | 77.4% | 90.3% | 93.5% | 1.90 |
| 39 | 27 | 85.2% | 92.6% | 96.3% | 1.37 |
| 40 | 37 | 81.1% | 91.9% | 94.6% | 1.57 |
| **整體** | **127** | **81.1%** | **92.1%** | **95.3%** | **1.61** |

## 完整模型歷史

### 模型比較表

| 模型名稱 | 架構 | Backbone | 損失函數 | 挖掘方式 | 同區域負樣本 | 圖片 | 測試準確率 | 測試損失 | 驗證損失 | N-Bench 平均 | A37 | A38 | A39 | A40 | 時間 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 預訓練 | - | DINOv2-B/14 | - | - | - | bbox | - | - | - | 29.48% | 28.12% | 35.48% | 29.63% | 24.32% | - |
| 20250812_152526 | 舊 | DINOv2-B/14 | Triplet | 預組三元組 | ❌ | bbox | 92.6% | 0.1659 | - | 48.25% | 50.00% | 51.61% | 48.15% | 43.24% | ~16h |
| 20251007_133126 | 舊 | DINOv2-B/14 | Triplet | 預組三元組 | ✅ | bbox | 88.8% | 0.2523 | - | 39.32% | 46.88% | 41.94% | 33.33% | 35.14% | ~16h |
| 20251008_094017 | 舊 | DINOv2-B/14 | Triplet | 預組三元組 | ✅ | bbox | 90.4% | 0.1636 | - | 40.19% | 37.50% | 48.39% | 37.04% | 37.84% | ~16h |
| 20251014_183603 | 舊 | DINOv2-B/14 | Triplet | 預組三元組 | ❌ | bbox | 92.8% | 0.1012 | - | 40.97% | 37.50% | 38.71% | 44.44% | 43.24% | ~16h |
| 預訓練 | - | DINOv2-B/14 | - | - | - | whole | - | - | - | 50.88% | 34.38% | 54.84% | 62.96% | 51.35% | - |
| 20251015_165008 | 舊 | DINOv2-B/14 | Triplet | 預組三元組 | ✅ | whole | 92.7% | 0.1330 | 0.1006 | 64.43% | 62.50% | 61.29% | 55.56% | 78.38% | ~16h |
| 20251016_133229 | 舊 | DINOv2-B/14 | Triplet | 預組三元組 | ❌ | whole | 97.9% | 0.0429 | - | 63.31% | 56.25% | 58.06% | 74.07% | 64.86% | ~16h |
| **20260308_110634** | **新** | **DINOv2-B/14 (timm 518)** | **Triplet** | **動態 (PML)** | **AreaAware 0.75** | whole | - | - | **0.1604** | **86.6%** | **93.8%** | **80.6%** | **85.2%** | **86.5%** | **~7.2h** |
| **20260306_233824** | **新** | **DINOv3-S+/16 (timm 512)** | **Triplet** | **動態 (PML)** | ❌ | whole | - | - | **0.1604** | **81.1%** | **81.2%** | **77.4%** | **85.2%** | **81.1%** | **~2.0h** |

### 欄位說明

| 欄位 | 說明 |
| --- | --- |
| 架構 | `舊` = 舊專案實作，`新` = 重構後的模組化架構 |
| Backbone | 特徵提取器（DINOv2-B/14、DINOv3-S+/16 等） |
| 損失函數 | 損失函數類型（Triplet、ArcFace、CosFace、Circle、Contrastive 等） |
| 挖掘方式 | 樣本挖掘策略：`預組三元組` = 固定三元組，`動態 (PML)` = MPerClassSampler |
| 同區域負樣本 | 是否限制負樣本來自同一地理區域（`AreaAware 0.75` = 75% 同區域） |
| 圖片 | `bbox` = EXIF 邊界框裁切，`whole` = 完整圖片 |
| 測試準確率 | 測試集準確率（僅舊架構，衡量 pos_dist < neg_dist） |
| 驗證損失 | 訓練期間最佳驗證損失 |
| N-Bench 平均 | N-Benchmark Top-1 準確率（區域 37-40 平均） |

### 架構差異

| 特性 | 舊架構 | 新架構 |
| --- | --- | --- |
| Dataset 輸出 | `(anchor, pos, neg)` - 3 張圖片 | `(image, label)` - 1 張圖片 |
| 三元組形成 | 訓練前預先組成 | 每批次動態挖掘 |
| 批次取樣器 | 隨機 | MPerClassSampler (m=2) |
| 損失函數 | 自訂 TripletLossWithMining | PML TripletMarginLoss |
| 每 Epoch 樣本數 | ~50,000 三元組 × 3 張圖片 | ~4,000 張圖片 |
| 訓練速度 | ~23 分鐘/epoch | ~1.5 分鐘/epoch |
| 同區域負樣本 | 已實作 | 已實作（AreaAwareSampler） |

> **N-Benchmark（最近鄰基準測試）**：在區域 37-40 中，跨 2022 與 2023 年比對珊瑚標本時，正確辨識的 Top-1 準確率。

## 專案結構

```
coral_models/
├── pyproject.toml                        # uv 環境（獨立）
├── extract_features.py                   # 特徵提取腳本
├── e3_01b_dinov2_vitb_best/              # 最高精度模型 (86.6%)
│   ├── e3_01b_same_area_neg_075.yaml     #   訓練配置
│   ├── README.md
│   └── README(zh-tw).md
├── dinov3_vitsplus_efficient/            # 最高效率模型 (81.1%)
│   ├── dinov3_vitsplus_tune_02_p2lr5_4ph.yaml  #   訓練配置
│   ├── README.md
│   └── README(zh-tw).md
├── 2022sample/                          # N-Benchmark 參考集圖片
└── 2023sample/                          # N-Benchmark 查詢集圖片
```

> 模型權重（`.pt` 檔案）與舊版模型託管於 [Hugging Face](https://huggingface.co/YuC13600/coral_models)。

## 授權條款

本專案採用 GPL-3.0 授權。

基於 Meta Platforms, Inc. 的 DINOv2 與 DINOv3（Apache License 2.0）。
