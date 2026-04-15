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
        boxes: Optional[np.ndarray],
        *,
        output_dir: PathLike,
        epochs: int = 20,
        lr: float = 1e-4,
        val_split: float = 0.1,
        seed: int = 42,
        text_prompt: str = "spleen",
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

        self.use_boxes = boxes is not None
        self.text_prompt = text_prompt
        print(f"[SAM trainer] prompt: "
              f"{'boxes' if self.use_boxes else f'text={text_prompt!r}'}")
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
        targets = masks.float().unsqueeze(1).to(self.device)

        if self.use_boxes:
            proc_kwargs = dict(input_boxes=boxes.unsqueeze(1).tolist())
        else:
            # Text prompt broadcast per image in the batch.
            proc_kwargs = dict(text=[self.text_prompt] * len(images_list))
        inputs = self.processor(
            images=images_list,
            return_tensors="pt",
            **proc_kwargs,
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


def _compute_boxes(
    images: np.ndarray,
    masks: np.ndarray,
    *,
    source: str = "none",
    unet_model=None,
) -> Optional[np.ndarray]:
    """Produce per-image bounding boxes for SAM3 training.

    source="none"  -> return None (text-only training, no box prompt)
    source="unet"  -> run UNet per image, fall back to GT if UNet misses
    source="gt"    -> tight box from the ground-truth mask (the 'cheating' baseline)
    """
    if source == "none":
        return None
    if source == "gt":
        boxes = np.ones((masks.shape[0], 4))
        for i in range(masks.shape[0]):
            boxes[i] = make_fake_box(masks[i])
        return boxes
    if source == "unet":
        if unet_model is None:
            raise ValueError("source='unet' requires unet_model")
        from impact_team_2.inference._inference_sam3 import _box_from_unet, _to_pil
        boxes = np.zeros((masks.shape[0], 4))
        misses = 0
        for i in range(masks.shape[0]):
            box = _box_from_unet(unet_model, _to_pil(images[i]), threshold=0.5)
            if box is None:
                h, w = images[i].shape[:2]
                box = np.array([0.0, 0.0, float(w), float(h)])
                misses += 1
            boxes[i] = box
        if misses:
            print(f"[SAM trainer] UNet missed on {misses}/{masks.shape[0]} images "
                  f"(fell back to full-image box — matches inference behavior)")
        return boxes
    raise ValueError(f"unknown box source: {source!r}")


def train_sam(
    images: np.ndarray,
    masks: np.ndarray,
    *,
    output_dir: PathLike,
    epochs: int = 20,
    lr: float = 1e-4,
    val_split: float = 0.1,
    seed: int = 42,
    box_source: str = "none",
    unet_model=None,
    text_prompt: str = "spleen",
) -> Path:
    """Fine-tune SAM3 on in-memory (images, masks).

    Args:
        images: uint8 array, shape (N, H, W, 3).
        masks:  bool/0-1 array, shape (N, H, W).
        output_dir: where checkpoints and TensorBoard logs are written.
        epochs, lr, val_split, seed: training hyperparameters.
        box_source: one of "none" (text-only), "unet" (UNet-predicted boxes,
            matches inference), or "gt" (boxes derived from GT masks — the
            "cheating" baseline, useful for upper-bound comparisons).
        unet_model: required when box_source="unet"; a loaded Keras UNet.
        text_prompt: text used as the prompt when box_source="none".

    Returns:
        Path to the best checkpoint (falls back to the last if no val split).
    """
    boxes = _compute_boxes(
        images, masks.astype(np.uint8),
        source=box_source, unet_model=unet_model,
    )
    trainer = SAMTrainer(
        images, masks, boxes,
        output_dir=output_dir, epochs=epochs, lr=lr,
        val_split=val_split, seed=seed, text_prompt=text_prompt,
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
    box_source: str = "none",
    unet_model=None,
    text_prompt: str = "spleen",
) -> Path:
    """CLI-friendly wrapper: load `images_in` / `masks_in` npz files, then train."""
    images = np.load(images_in)["images"]
    masks = np.load(masks_in)["masks"]
    return train_sam(
        images, masks,
        output_dir=model_out, epochs=epochs, lr=lr,
        val_split=val_split, seed=seed,
        box_source=box_source, unet_model=unet_model, text_prompt=text_prompt,
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
