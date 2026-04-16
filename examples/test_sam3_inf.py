#!/usr/bin/env python3
"""Detect-then-segment evaluation harness for SAM3 on spleen ultrasound.

Two-stage cascade: the vendor Keras UNet++ (INIA) predicts a coarse segmentation
and emits a bounding box, SAM3 then refines inside that box using a text prompt
to produce the final mask. The script runs the pipeline over a seeded held-out
val split and reports per-image dice + summary stats (mean, median, min, max,
count > 0.5, count > 0.3).

By default the baseline `facebook/sam3` weights are used. Passing `--weights`
loads a fine-tuned SAM3 checkpoint (from `test_sam3_train.py`) on top, so you
can diff baseline vs. fine-tuned numbers. Passing `--train` first retrains the
UNet from scratch before evaluation.

Side effects under `runs/` (or `--save-overlays-dir`):
    sam3_{baseline|finetuned}_val_NNN_dice0.XX[_score0.XX].png  — 4-panel overlay
        (image | GT | pred | diff) per selected val image. Selection is
        controlled by `--save-overlays-n` (int N / "all" / "worst:K" / "best:K").
    sam3_{baseline|finetuned}_<stem>_overlay.png  — each extra positional image

Usage:
    python examples/test_sam3_inf.py                              # baseline eval
    python examples/test_sam3_inf.py --weights <.safetensors>     # fine-tuned eval
    python examples/test_sam3_inf.py --unet path/to/unet.h5       # custom UNet weights
    python examples/test_sam3_inf.py --train                      # retrain UNet first
    python examples/test_sam3_inf.py extra/image.png              # also overlay these
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split

_repo = Path(__file__).resolve().parent.parent
from _data import load_spleen_data

from impact_team_2.inference._inference_sam3 import build_predictor, load_unet_weights
from impact_team_2.visual import dice_score, resize_mask, save_overlay, resolve_save_indices

try:
    from impact_team_2.vendor.team_one.INIA import (  # noqa: E402
        fit, load_data, get_bboxes, plot_history, plot_predictions,
    )
except ImportError as e:
    raise ImportError(
        "The SAM3 auto-bbox path requires the `unet` extra. Install it with:\n"
        "    pip install -e \".[unet]\"\n"
        f"(original error: {e})"
    ) from e


TEXT_PROMPT = "spleen"
DEFAULT_UNET = _repo / "checkpoints" / "best_unetp.weights.h5"
DATA_DIR = _repo / "datasets" / "ultrasound_spleen"
OUT_DIR = _repo / "runs"


def _parse_n(value: str):
    """Let --save-overlays-n accept either an int or a string selector."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _extract_best(result: dict) -> tuple[Optional[np.ndarray], Optional[float]]:
    if result.get("masks") is None or result.get("scores") is None:
        return None, None
    scores = result["scores"]
    idx = int(scores.argmax())
    return result["masks"][idx].astype(bool), float(scores[idx])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*",
                        help="Extra image paths to overlay (no dice for these)")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Fine-tuned SAM3 .safetensors checkpoint")
    parser.add_argument("--unet", type=Path, default=DEFAULT_UNET,
                        help="UNet++ weights for automatic bbox generation")
    parser.add_argument("--no-unet", action="store_true",
                        help="Skip the UNet bbox step; evaluate SAM3 on text prompt alone "
                             "(use this to evaluate a text-only fine-tuned model)")
    parser.add_argument("--train", action="store_true",
                        help="Retrain the UNet before evaluation")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction held out for evaluation (must match training)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-overlays-dir", type=Path, default=None,
                        help="Directory to write per-image overlay PNGs "
                             "(image|GT|pred|diff). Default: runs/.")
    parser.add_argument("--save-overlays-n", default="3",
                        help="Which overlays to save: int N (first N), 'all', "
                             "'worst:K', or 'best:K'. Default: 3 (back-compat with --n-overlays).")
    parser.add_argument("--n-overlays", type=int, default=None,
                        help="[deprecated] alias for --save-overlays-n.")
    args = parser.parse_args()

    # --n-overlays back-compat
    if args.n_overlays is not None:
        args.save_overlays_n = str(args.n_overlays)

    # --- data --------------------------------------------------------------
    print("[eval] loading spleen data")
    images, masks = load_spleen_data()
    idx = np.arange(images.shape[0])
    _, val_idx = train_test_split(idx, test_size=args.val_split, random_state=args.seed)
    val_images = images[val_idx]
    val_masks = masks[val_idx]
    print(f"[eval] val split: {len(val_idx)} / {images.shape[0]}")

    # --- UNet --------------------------------------------------------------
    if args.no_unet:
        if args.train:
            raise ValueError("--train and --no-unet are mutually exclusive")
        unet_model = None
        print("[UNet] skipped (--no-unet): SAM3 will use the text prompt alone")
    elif args.train:
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
                f"Pass --train to train a fresh UNet, --unet to point at an existing one, "
                f"or --no-unet to evaluate text-only."
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
    buf: list[dict] = []  # keep (pred_mask, score, image) per val image for overlay
    for i in range(len(val_images)):
        result = predictor(val_images[i], text_prompt=TEXT_PROMPT, threshold=args.threshold)
        best, best_score = _extract_best(result)
        if best is None:
            dice_scores.append(0.0)
            buf.append({"pred": None, "score": None, "img": result.get("image")})
            print(f"  val {i:3d}: dice=0.0000  (no detections)")
            continue
        gt_resized = resize_mask(val_masks[i], best.shape)
        d = dice_score(best, gt_resized)
        dice_scores.append(d)
        buf.append({"pred": best, "score": best_score, "img": result.get("image")})
        print(f"  val {i:3d}: dice={d:.4f}  detections={result['num_detections']}")

    # --- write overlays ---------------------------------------------------
    overlays_dir = args.save_overlays_dir if args.save_overlays_dir else OUT_DIR
    try:
        save_indices = resolve_save_indices(dice_scores, _parse_n(args.save_overlays_n))
    except ValueError as e:
        print(f"[eval] skipping overlays: {e}")
        save_indices = []
    for i in save_indices:
        b = buf[i]
        parts = [f"sam3_{tag}_val_{i:03d}", f"dice{dice_scores[i]:.3f}"]
        if b["score"] is not None:
            parts.append(f"score{b['score']:.3f}")
        out_path = overlays_dir / ("_".join(parts) + ".png")
        save_overlay(
            b["img"] if b["img"] is not None else val_images[i],
            val_masks[i],
            b["pred"],
            out_path,
            dice=dice_scores[i],
            score=b["score"],
            title=f"SAM3/{tag} · val[{i}]",
        )
    if save_indices:
        print(f"[eval] wrote {len(save_indices)} overlay(s) to {overlays_dir}")

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
        best, best_score = _extract_best(result)
        save_overlay(
            result["image"],
            None,
            best,
            overlays_dir / f"sam3_{tag}_{img_path.stem}_overlay.png",
            score=best_score,
            title=f"SAM3/{tag} · {img_path.name}",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
