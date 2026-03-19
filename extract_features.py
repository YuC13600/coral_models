"""Standalone feature extraction for coral re-identification models.

Reconstructs the model architecture from checkpoint metadata (or a YAML config
as fallback) and loads weights without depending on the coral_reid package.

Usage:
    # Extract features from a directory of images
    uv run python extract_features.py \
        --model e3_01b_dinov2_vitb_best/best_model_20260308_110634.pt \
        --input /path/to/images \
        --output features.h5

    # Extract features for N-Benchmark (by area)
    uv run python extract_features.py \
        --model e3_01b_dinov2_vitb_best/best_model_20260308_110634.pt \
        --input /path/to/2022sample \
        --areas 37 38 39 40 \
        --output features/

    # Single image embedding (prints to stdout)
    uv run python extract_features.py \
        --model e3_01b_dinov2_vitb_best/best_model_20260308_110634.pt \
        --input /path/to/single_image.jpg
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Model configuration parsed from YAML."""

    # Backbone
    backbone_variant: str
    img_size: int
    backbone_output_dim: int

    # Head
    hidden_dim: int
    output_dim: int
    dropout: float
    use_batchnorm: bool

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        """Create config from a dict (embedded in checkpoint)."""
        return cls(
            backbone_variant=d["backbone_variant"],
            img_size=d.get("img_size", 224),
            backbone_output_dim=d["backbone_output_dim"],
            hidden_dim=d["hidden_dim"],
            output_dim=d["output_dim"],
            dropout=d.get("dropout", 0.3),
            use_batchnorm=d.get("use_batchnorm", True),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        with open(path) as f:
            cfg = yaml.safe_load(f)

        backbone = cfg["backbone"]
        head = cfg["head"]

        return cls(
            backbone_variant=backbone["variant"],
            img_size=backbone.get("img_size", 224),
            backbone_output_dim=backbone["output_dim"],
            hidden_dim=head["hidden_dim"],
            output_dim=head["output_dim"],
            dropout=head.get("dropout", 0.3),
            use_batchnorm=head.get("use_batchnorm", True),
        )


# ---------------------------------------------------------------------------
# Model Architecture (standalone reconstruction)
# ---------------------------------------------------------------------------


class MLPHead(nn.Module):
    """MLP projection head with L2 normalization.

    Architecture:
        BatchNorm1d → Dropout(0.2)
        → Linear → ReLU → Dropout → Linear → [BatchNorm1d]
        → L2 Normalize
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        self.feature_processor = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(p=0.2),
        )

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(output_dim))

        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_processor(x)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)


class CoralReIDModel(nn.Module):
    """Coral re-identification model: timm backbone + MLP head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # Backbone: timm model with classification head removed
        self.backbone = timm.create_model(
            config.backbone_variant,
            pretrained=False,  # weights come from checkpoint
            num_classes=0,
            img_size=config.img_size,
        )

        self.head = MLPHead(
            input_dim=config.backbone_output_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            dropout=config.dropout,
            use_batchnorm=config.use_batchnorm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    config_path: str | Path | None = None,
) -> tuple[CoralReIDModel, ModelConfig]:
    """Load model from checkpoint file.

    Model config is read from the checkpoint's ``model_config`` key.
    If the checkpoint doesn't contain it, ``config_path`` (YAML) is used
    as a fallback.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load the model on.
        config_path: Optional path to a YAML config (fallback).

    Returns:
        Tuple of (model, config).
    """
    # Checkpoint is a dict with "model_state_dict" key
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Resolve config: checkpoint-embedded > YAML fallback
    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        config = ModelConfig.from_dict(checkpoint["model_config"])
    elif config_path is not None:
        config = ModelConfig.from_yaml(config_path)
    else:
        raise ValueError(
            "Checkpoint does not contain model_config and no --config provided. "
            "Use embed_config.py to add config to the checkpoint, or pass --config."
        )

    model = CoralReIDModel(config)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Fallback: raw state_dict
        state_dict = checkpoint

    # Map keys: original uses "backbone.model.*", timm direct uses "backbone.*"
    mapped_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("backbone.model."):
            new_key = key.replace("backbone.model.", "backbone.", 1)
        else:
            new_key = key
        mapped_state_dict[new_key] = value

    model.load_state_dict(mapped_state_dict)
    model.to(device)
    model.eval()

    logger.info(
        f"Loaded model: {config.backbone_variant} "
        f"({config.img_size}px, {config.output_dim}d embedding)"
    )
    return model, config


# ---------------------------------------------------------------------------
# Inference Transforms
# ---------------------------------------------------------------------------


def get_inference_transforms(image_size: int) -> transforms.Compose:
    """Create inference transforms matching training pipeline."""
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_single(
    model: CoralReIDModel,
    img_path: str | Path,
    transform: transforms.Compose,
    device: str | torch.device,
) -> np.ndarray | None:
    """Extract feature embedding from a single image."""
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        embedding = model(tensor)
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        logger.warning(f"Failed to process {img_path}: {e}")
        return None


@torch.no_grad()
def extract_directory(
    model: CoralReIDModel,
    directory: str | Path,
    transform: transforms.Compose,
    device: str | torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[str]]:
    """Extract features from all images in a directory.

    Returns:
        Tuple of (features array [N, D], list of coral names).
    """
    directory = Path(directory)
    image_files = sorted(
        f
        for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not image_files:
        logger.warning(f"No images found in {directory}")
        return np.array([]), []

    features_list: list[np.ndarray] = []
    coral_names: list[str] = []

    for i in tqdm(range(0, len(image_files), batch_size), desc=str(directory)):
        batch_files = image_files[i : i + batch_size]
        batch_tensors: list[torch.Tensor] = []
        batch_names: list[str] = []

        for fname in batch_files:
            try:
                img = Image.open(directory / fname).convert("RGB")
                batch_tensors.append(transform(img))
                batch_names.append(os.path.splitext(fname)[0])
            except Exception as e:
                logger.warning(f"Skipping {fname}: {e}")

        if batch_tensors:
            batch = torch.stack(batch_tensors).to(device)
            feats = model(batch).cpu().numpy()
            features_list.append(feats)
            coral_names.extend(batch_names)

    if features_list:
        features = np.concatenate(features_list, axis=0)
    else:
        features = np.array([])

    return features, coral_names


def save_features_h5(
    path: str | Path,
    features: np.ndarray,
    coral_names: list[str],
    metadata: dict[str, str | int | float] | None = None,
) -> None:
    """Save features to HDF5 file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=features)
        f.create_dataset(
            "coral_names",
            data=[name.encode("utf-8") for name in coral_names],
        )
        f.attrs["feature_dim"] = features.shape[1] if len(features.shape) > 1 else 0
        f.attrs["num_samples"] = features.shape[0]

        if metadata:
            for key, value in metadata.items():
                if value is not None:
                    f.attrs[key] = value

    logger.info(f"Saved {len(coral_names)} features to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone feature extraction for coral re-identification models",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (optional if config is embedded in checkpoint)",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to image file or directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (.h5 file or directory for area mode)",
    )
    parser.add_argument(
        "--areas",
        nargs="+",
        default=None,
        help="Area IDs for N-Benchmark extraction (e.g., 37 38 39 40)",
    )
    parser.add_argument(
        "--year",
        default=None,
        help="Year label for area mode filenames (e.g., 2022)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for extraction (default: 32)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    # Load model
    model, config = load_model(args.model, args.device, config_path=args.config)
    transform = get_inference_transforms(config.img_size)

    # --- Single image mode ---
    if input_path.is_file():
        embedding = extract_single(model, input_path, transform, args.device)
        if embedding is not None:
            print(f"Image: {input_path.name}")
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
            if args.output:
                np.save(args.output, embedding)
                logger.info(f"Saved embedding to {args.output}")
            else:
                print(f"Embedding: {embedding[:8]}... (first 8 dims)")
        return

    # --- Area mode (N-Benchmark style) ---
    if args.areas:
        output_dir = Path(args.output) if args.output else Path("features")
        output_dir.mkdir(parents=True, exist_ok=True)

        for area_id in args.areas:
            area_dir = input_path / area_id
            if not area_dir.exists():
                logger.warning(f"Area directory not found: {area_dir}")
                continue

            features, names = extract_directory(
                model, area_dir, transform, args.device, args.batch_size,
            )
            if len(features) > 0:
                if args.year:
                    out_path = output_dir / f"features_{args.year}_{area_id}_whole.h5"
                else:
                    out_path = output_dir / f"features_{area_id}_whole.h5"
                save_features_h5(
                    out_path,
                    features,
                    names,
                    {"area_id": area_id, "source_dir": str(area_dir)},
                )
        return

    # --- Directory mode ---
    if input_path.is_dir():
        features, names = extract_directory(
            model, input_path, transform, args.device, args.batch_size,
        )
        if len(features) > 0:
            output_path = args.output or "features.h5"
            save_features_h5(
                output_path,
                features,
                names,
                {"source_dir": str(input_path)},
            )
        else:
            logger.error("No features extracted")
        return

    logger.error(f"Input path not found: {input_path}")


if __name__ == "__main__":
    main()
