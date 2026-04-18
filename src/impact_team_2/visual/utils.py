"""Pure numpy/PIL mask helpers shared across the package.

Keep this module dependency-light — numpy + PIL only. No matplotlib, tqdm,
torch, or tensorflow imports. This lets `api.py` and `overlays.py` pull the
helpers without dragging heavy optional deps into their import graphs.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PIL import Image as PILImage


def best_mask(result: dict) -> Optional[np.ndarray]:
    """Return the highest-scoring mask from a predictor result as a bool array.

    `result` is the dict returned by MedSAM3 / SAM3 predictors — ``masks`` of
    shape (N, H, W) and ``scores`` of shape (N,). Returns None if either is
    missing, e.g. when the model produced no detections.
    """
    if result.get("masks") is None or result.get("scores") is None:
        return None
    return result["masks"][int(result["scores"].argmax())].astype(bool)


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Binary dice coefficient. Operates on bool / 0-1 arrays of the same shape."""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return 0.0
    return float(2 * intersection / (denom + 1e-8))


def resize_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Nearest-neighbor resize a boolean/0-1 mask to `shape=(H, W, ...)`.

    Only the first two entries of `shape` are used — this matches
    `ndarray.shape` so callers can pass it directly. Always returns a bool
    array. If the mask already has the target H/W, it is returned as bool
    without a PIL round-trip.
    """
    target = (shape[0], shape[1])
    if mask.shape == target:
        return mask.astype(bool)
    return np.array(
        PILImage.fromarray(mask.astype(np.uint8)).resize(
            (target[1], target[0]), PILImage.NEAREST
        ),
        dtype=bool,
    )


def summarize_dice(
    dice_scores: Sequence[float],
    scores: Optional[Sequence[float]] = None,
) -> dict:
    """Summary dict over a list of per-image dice scores.

    `scores` is the optional flat list of per-detection confidence scores
    across all evaluated images — when provided, the summary also includes
    `score_min`, `score_max`, `score_mean`.
    """
    out: dict = {
        "n": len(dice_scores),
        "mean_dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "max_dice": float(np.max(dice_scores)) if dice_scores else 0.0,
        "min_dice": float(np.min(dice_scores)) if dice_scores else 0.0,
        "dice_gt_0.5": int(sum(1 for d in dice_scores if d > 0.5)),
        "dice_gt_0.3": int(sum(1 for d in dice_scores if d > 0.3)),
    }
    if scores is not None and len(scores) > 0:
        out["score_min"] = float(np.min(scores))
        out["score_max"] = float(np.max(scores))
        out["score_mean"] = float(np.mean(scores))
    return out


__all__ = ["best_mask", "dice_score", "resize_mask", "summarize_dice"]
