#!/usr/bin/env python3
"""SAM3 inference + evaluation on the spleen ultrasound val split.

Always evaluates SAM3 (optionally with a fine-tuned `--weights` checkpoint)
against the held-out val split and reports per-image dice + summary. Bounding
boxes are generated automatically by the vendor UNet (INIA). Overlays for a
handful of val images are saved under `runs/`.

Usage:
    python examples/test_sam3_inf.py
    python examples/test_sam3_inf.py --weights runs/sam3_finetune/sam3_finetuned_weights.safetensors
    python examples/test_sam3_inf.py --unet path/to/unet.h5
    python examples/test_sam3_inf.py --train                       # retrain UNet first
    python examples/test_sam3_inf.py extra/image.png               # also overlay these
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from sklearn.model_selection import train_test_split

_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import load_spleen_data  # noqa: E402

from impact_team_2.inference._inference_sam3 import build_predictor, load_unet_weights  # noqa: E402
from impact_team_2.vendor.team_one.INIA import (  # noqa: E402
    fit, load_data, get_bboxes, plot_history, plot_predictions,
)


TEXT_PROMPT = "spleen ,organ, spleen organ in ultrasound, dark oval region ,hypoechoic mass"
DEFAULT_UNET = _repo / "checkpoints" / "best_unetp.weights.h5"
DATA_DIR = _repo / "datasets" / "ultrasound_spleen"
OUT_DIR = _repo / "runs"


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 0.0
    return float(2 * np.logical_and(pred, gt).sum() / denom)


def _save_overlay(result: dict, out_path: Path) -> None:
    if result["masks"] is None:
        return
    best = result["masks"][int(result["scores"].argmax())]
    base = np.array(result["image"].convert("RGBA"))
    overlay = np.zeros_like(base)
    overlay[best] = [171, 71, 188, 140]
    alpha = overlay[:, :, 3:4] / 255.0
    base[:, :, :3] = (
        base[:, :, :3] * (1 - alpha) + overlay[:, :, :3] * alpha
    ).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(base, mode="RGBA").convert("RGB").save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*",
                        help="Extra image paths to overlay (no dice for these)")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Fine-tuned SAM3 .safetensors checkpoint")
    parser.add_argument("--unet", type=Path, default=DEFAULT_UNET,
                        help="UNet++ weights for automatic bbox generation")
    parser.add_argument("--train", action="store_true",
                        help="Retrain the UNet before evaluation")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction held out for evaluation (must match training)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-overlays", type=int, default=3,
                        help="How many val-set overlays to save")
    args = parser.parse_args()

    # --- data --------------------------------------------------------------
    print("[eval] loading spleen data")
    images, masks = load_spleen_data()
    idx = np.arange(images.shape[0])
    _, val_idx = train_test_split(idx, test_size=args.val_split, random_state=args.seed)
    val_images = images[val_idx]
    val_masks = masks[val_idx]
    print(f"[eval] val split: {len(val_idx)} / {images.shape[0]}")

    # --- UNet --------------------------------------------------------------
    if args.train:
        print("[UNet] retraining from scratch")
        X_train, y_train, X_test, y_test = load_data(
            images_path=str(DATA_DIR / "images.npz"),
            masks_path=str(DATA_DIR / "masks.npz"),
        )
        unet_model, history = fit("unet++", X_train, y_train)
        plot_history(history, "unet++")
        plot_predictions(unet_model, X_test, y_test, n=5)
        (_repo / "checkpoints").mkdir(exist_ok=True)
        unet_model.save_weights(str(DEFAULT_UNET))
        print(f"[UNet] weights saved -> {DEFAULT_UNET}")
    else:
        if not args.unet.exists():
            raise FileNotFoundError(
                f"UNet weights not found at {args.unet}. "
                f"Pass --train to train a fresh UNet, or --unet to point at an existing one."
            )
        unet_model = load_unet_weights(args.unet)

    # --- SAM3 predictor ----------------------------------------------------
    predictor = build_predictor(
        unet_model=unet_model,
        weights_path=args.weights,
    )
    tag = "finetuned" if args.weights else "baseline"
    print(f"[SAM3] mode: {tag}"
          + (f" (weights={args.weights})" if args.weights else ""))

    # --- evaluate on val split --------------------------------------------
    dice_scores: list[float] = []
    for i in range(len(val_images)):
        result = predictor(val_images[i], text_prompt=TEXT_PROMPT, threshold=args.threshold)
        if result["masks"] is None or result["scores"] is None:
            dice_scores.append(0.0)
            continue
        best = result["masks"][int(result["scores"].argmax())].astype(bool)
        gt = val_masks[i].astype(bool)
        if best.shape != gt.shape:
            gt_resized = np.array(
                PILImage.fromarray(gt.astype(np.uint8)).resize(
                    (best.shape[1], best.shape[0]), PILImage.NEAREST
                ),
                dtype=bool,
            )
        else:
            gt_resized = gt
        d = _dice(best, gt_resized)
        dice_scores.append(d)

        if i < args.n_overlays:
            out_path = OUT_DIR / f"sam3_{tag}_val_{i:03d}_overlay.png"
            _save_overlay(result, out_path)
            print(f"  val {i:3d}: dice={d:.4f}  detections={result['num_detections']}"
                  f"  overlay={out_path.name}")
        else:
            print(f"  val {i:3d}: dice={d:.4f}  detections={result['num_detections']}")

    ds = np.array(dice_scores)
    print(f"\n[eval] summary ({tag})")
    print(f"  n          = {len(ds)}")
    print(f"  mean dice  = {ds.mean():.4f}")
    print(f"  median     = {np.median(ds):.4f}")
    print(f"  min / max  = {ds.min():.4f} / {ds.max():.4f}")
    print(f"  dice>0.5   = {int((ds > 0.5).sum())}")
    print(f"  dice>0.3   = {int((ds > 0.3).sum())}")

    # --- extra user-provided images (no GT) --------------------------------
    for img_path in (Path(p) for p in args.images):
        result = predictor(img_path, text_prompt=TEXT_PROMPT, threshold=args.threshold)
        print(f"[extra] {img_path.name} — detections: {result['num_detections']}")
        _save_overlay(result, OUT_DIR / f"sam3_{tag}_{img_path.stem}_overlay.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
