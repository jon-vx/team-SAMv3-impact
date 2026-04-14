"""MedSAM3 LoRA fine-tuning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pycocotools.mask as mask_utils
import yaml
from PIL import Image as PILImage
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split

from impact_team_2.vendor.medsam3.lora_layers import load_lora_weights
from impact_team_2.vendor.medsam3.train_sam3_lora_native import SAM3TrainerNative


# ---------------------------------------------------------------------------
# Config dataclasses 
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: Optional[list] = None
    apply_to_vision_encoder: bool = True
    apply_to_text_encoder: bool = True
    apply_to_geometry_encoder: bool = True
    apply_to_detr_encoder: bool = True
    apply_to_detr_decoder: bool = True
    apply_to_mask_decoder: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_workers: int = 0


# ---------------------------------------------------------------------------
# COCO export
# ---------------------------------------------------------------------------

def _write_coco_split(
    images: np.ndarray,
    masks: np.ndarray,
    indices: Sequence[int],
    split_dir: Path,
    category_name: str,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": category_name}],
    }

    ann_id = 0
    for new_id, idx in enumerate(indices):
        filename = f"img_{int(idx):04d}.png"
        PILImage.fromarray(images[idx]).save(split_dir / filename)

        h, w = images[idx].shape[:2]
        coco["images"].append(
            {"id": new_id, "file_name": filename, "width": int(w), "height": int(h)}
        )

        mask = masks[idx].astype(np.uint8)
        if mask.sum() == 0:
            continue

        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min()), int(ys.max() - ys.min())]

        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("utf-8")

        coco["annotations"].append({
            "id": ann_id,
            "image_id": new_id,
            "category_id": 1,
            "segmentation": rle,
            "bbox": bbox,
            "area": int(mask.sum()),
            "iscrowd": 0,
        })
        ann_id += 1

    with open(split_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)


def export_coco_dataset(
    images: np.ndarray,
    masks: np.ndarray,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    dataset_dir: Path | str,
    category_name: str = "object",
) -> Path:
    dataset_dir = Path(dataset_dir)
    _write_coco_split(images, masks, train_idx, dataset_dir / "train", category_name)
    _write_coco_split(images, masks, val_idx, dataset_dir / "valid", category_name)
    return dataset_dir


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def _resolve_pretrained_lora(pretrained_lora: Optional[str]) -> Optional[Path]:
    if not pretrained_lora:
        return None
    path = Path(pretrained_lora)
    if path.is_file():
        return path
    snapshot_dir = Path(snapshot_download(repo_id=pretrained_lora))
    weights = next(snapshot_dir.glob("*.pt"), None)
    if weights is None:
        raise FileNotFoundError(
            f"No .pt file found in MedSAM3 LoRA snapshot at {snapshot_dir}"
        )
    return weights


def train_medsam3(
    images: np.ndarray,
    masks: np.ndarray,
    output_dir: Path | str,
    *,
    category: str = "object",
    val_split: float = 0.2,
    seed: int = 42,
    lora_config: Optional[LoRAConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    pretrained_lora: Optional[str] = "lal-Joey/MedSAM3_v1",
) -> Path:
    """Fine-tune MedSAM3 LoRA on an in-memory (images, masks) pair.

    Args:
        images: uint8 array, shape (N, H, W, 3).
        masks:  bool/0-1 array, shape (N, H, W).
        output_dir: where the COCO dataset, config, and LoRA checkpoints are written.
        category: text prompt / class name written into the COCO categories.
        val_split: fraction of `images` held out for validation loss.
        seed: random seed for the train/val split only.
        lora_config / training_config: optional overrides; defaults match the notebook.
        pretrained_lora: HF repo id or local `.pt` path used as a warm start.
            Pass `None` to skip the warm start (train LoRA from scratch).

    Returns:
        Path to `best_lora_weights.pt` (the same checkpoint format the inference
        path in `impact_team_2.inference` consumes).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_config = lora_config or LoRAConfig()
    training_config = training_config or TrainingConfig()

    # 1. train/val split + COCO export
    n = images.shape[0]
    train_idx, val_idx = train_test_split(
        range(n), test_size=val_split, random_state=seed
    )
    dataset_dir = output_dir / "dataset"
    export_coco_dataset(images, masks, train_idx, val_idx, dataset_dir, category)

    # 2. config yaml that SAM3TrainerNative consumes
    config = {
        "lora": asdict(lora_config),
        "training": {
            **asdict(training_config),
            "data_dir": str(dataset_dir),
        },
        "output": {"output_dir": str(output_dir)},
    }
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    # 3. build trainer + warm-start LoRA weights (notebook's monkey-patch)
    trainer = SAM3TrainerNative(str(config_path), multi_gpu=False)
    warm_start = _resolve_pretrained_lora(pretrained_lora)
    if warm_start is not None:
        load_lora_weights(trainer._unwrapped_model, str(warm_start))
        print(f"Loaded pretrained MedSAM3 LoRA weights from {warm_start}")

    # 4. train
    trainer.train()
    return output_dir / "best_lora_weights.pt"
