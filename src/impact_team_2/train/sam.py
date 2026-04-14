"""SAM3 fine-tuning on (images, masks) pairs."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from transformers import Sam3Model, Sam3Processor

from impact_team_2.train.data import SAM3Dataset, make_fake_box

PathLike = Union[str, Path]
_CHECKPOINT_NAME = "sam3_finetuned_weights.safetensors"
_HF_MODEL_ID = "facebook/sam3"


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_metrics(device: torch.device) -> MetricCollection:
    return MetricCollection({
        "dice": BinaryF1Score(),
        "iou": BinaryJaccardIndex(),
        "loss": MeanMetric(),
    }).to(device)


class SAMTrainer:
    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
        *,
        output_dir: PathLike,
        epochs: int = 20,
        lr: float = 1e-4,
        val_split: float = 0.1,
        seed: int = 42,
        device: Optional[torch.device] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.device = device or _pick_device()
        print(f"[SAM trainer] device: {self.device}")

        self.processor = Sam3Processor.from_pretrained(_HF_MODEL_ID)
        self.model = Sam3Model.from_pretrained(_HF_MODEL_ID).to(self.device)

        # Freeze everything except the mask decoder.
        for name, param in self.model.named_parameters():
            param.requires_grad = "mask_decoder" in name
        self._trainable_keys = {
            n for n, p in self.model.named_parameters() if p.requires_grad
        }
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[SAM trainer] trainable params: {n_trainable:,}")

        self.optimizer = Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

        dataset = SAM3Dataset(images, masks, boxes)
        self.train_loader, self.val_loader = self._split(dataset, val_split, seed)

        self.train_metrics = _build_metrics(self.device)
        self.val_metrics = _build_metrics(self.device)

        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb"))
        self.global_step = 0
        self.best_val_dice = -1.0

    @staticmethod
    def _split(dataset, val_split: float, seed: int):
        n = len(dataset)
        n_val = max(1, int(round(n * val_split))) if val_split > 0 else 0
        if n_val == 0 or n_val >= n:
            return DataLoader(dataset, batch_size=1, shuffle=True), None
        idx = np.arange(n)
        train_idx, val_idx = train_test_split(idx, test_size=n_val, random_state=seed)
        return (
            DataLoader(Subset(dataset, train_idx.tolist()), batch_size=1, shuffle=True),
            DataLoader(Subset(dataset, val_idx.tolist()), batch_size=1, shuffle=False),
        )

    def _forward(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        images, masks, boxes = batch
        images_list = [img.numpy() for img in images]
        boxes_list = boxes.unsqueeze(1).tolist()
        targets = masks.float().unsqueeze(1).to(self.device)

        inputs = self.processor(
            images=images_list,
            input_boxes=boxes_list,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)

        logits = outputs.pred_masks[:, :1, :, :]
        logits = F.interpolate(
            logits, size=targets.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits, targets

    def _update_metrics(self, metrics, logits, targets, loss):
        probs = torch.sigmoid(logits).detach()
        target_int = targets.long()
        metrics["dice"].update(probs, target_int)
        metrics["iou"].update(probs, target_int)
        metrics["loss"].update(loss.detach())

    def train_one_epoch(self) -> dict:
        self.model.train()
        self.train_metrics.reset()
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            logits, targets = self._forward(batch)
            loss = self.loss_fn(logits, targets)
            loss.backward()
            self.optimizer.step()
            self._update_metrics(self.train_metrics, logits, targets, loss)
            self.writer.add_scalar("train/step_loss", loss.item(), self.global_step)
            self.global_step += 1
        return {k: v.item() for k, v in self.train_metrics.compute().items()}

    @torch.inference_mode()
    def eval_one_epoch(self) -> dict:
        if self.val_loader is None:
            return {}
        self.model.eval()
        self.val_metrics.reset()
        for batch in self.val_loader:
            logits, targets = self._forward(batch)
            loss = self.loss_fn(logits, targets)
            self._update_metrics(self.val_metrics, logits, targets, loss)
        return {k: v.item() for k, v in self.val_metrics.compute().items()}

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        full = self.model.state_dict()
        trainable = {
            k: v.detach().cpu().contiguous()
            for k, v in full.items()
            if k in self._trainable_keys
        }
        save_file(trainable, str(path))
        size_mb = path.stat().st_size / 1e6
        print(f"[SAM trainer] saved {len(trainable)} tensors "
              f"({size_mb:.1f} MB) -> {path}")
        return path

    def fit(self) -> Path:
        best_path = self.output_dir / _CHECKPOINT_NAME
        last_path = self.output_dir / ("last_" + _CHECKPOINT_NAME)
        try:
            for epoch in range(1, self.epochs + 1):
                train_stats = self.train_one_epoch()
                val_stats = self.eval_one_epoch()

                for k, v in train_stats.items():
                    self.writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in val_stats.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

                msg = f"[epoch {epoch}/{self.epochs}] " + ", ".join(
                    f"train_{k}={v:.4f}" for k, v in train_stats.items()
                )
                if val_stats:
                    msg += " | " + ", ".join(
                        f"val_{k}={v:.4f}" for k, v in val_stats.items()
                    )
                print(msg)

                self.save(last_path)
                val_dice = val_stats.get("dice", float("nan"))
                if val_stats and val_dice > self.best_val_dice:
                    self.best_val_dice = val_dice
                    self.save(best_path)

            # No val split -> promote the last checkpoint to the canonical slot.
            if self.val_loader is None:
                import shutil
                shutil.copyfile(last_path, best_path)
        finally:
            self.writer.flush()
            self.writer.close()

        return best_path


def _compute_boxes(masks: np.ndarray) -> np.ndarray:
    boxes = np.ones((masks.shape[0], 4))
    for i in range(masks.shape[0]):
        boxes[i] = make_fake_box(masks[i])
    return boxes


def train_sam(
    images: np.ndarray,
    masks: np.ndarray,
    *,
    output_dir: PathLike,
    epochs: int = 20,
    lr: float = 1e-4,
    val_split: float = 0.1,
    seed: int = 42,
) -> Path:
    """Fine-tune SAM3 on in-memory (images, masks).

    Args:
        images: uint8 array, shape (N, H, W, 3).
        masks:  bool/0-1 array, shape (N, H, W).
        output_dir: where checkpoints and TensorBoard logs are written.
        epochs, lr, val_split, seed: training hyperparameters.

    Returns:
        Path to the best checkpoint (falls back to the last if no val split).
    """
    boxes = _compute_boxes(masks.astype(np.uint8))
    trainer = SAMTrainer(
        images, masks, boxes,
        output_dir=output_dir, epochs=epochs, lr=lr,
        val_split=val_split, seed=seed,
    )
    return trainer.fit()


def train_sam_from_files(
    images_in: PathLike,
    masks_in: PathLike,
    model_out: PathLike,
    *,
    epochs: int = 20,
    lr: float = 1e-4,
    val_split: float = 0.1,
    seed: int = 42,
) -> Path:
    """CLI-friendly wrapper: load `images_in` / `masks_in` npz files, then train."""
    images = np.load(images_in)["images"]
    masks = np.load(masks_in)["masks"]
    return train_sam(
        images, masks,
        output_dir=model_out, epochs=epochs, lr=lr,
        val_split=val_split, seed=seed,
    )


if __name__ == "__main__":
    parser = ArgumentParser(prog="train/sam.py")
    parser.add_argument("images_in")
    parser.add_argument("masks_in")
    parser.add_argument("model_out")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train_sam_from_files(
        images_in=args.images_in,
        masks_in=args.masks_in,
        model_out=args.model_out,
        epochs=args.epochs,
        val_split=args.val_split,
        lr=args.lr,
    )
