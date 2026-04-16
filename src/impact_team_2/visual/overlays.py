"""Prediction overlay helper.

Writes a 4-panel PNG — `image | GT overlay | pred overlay | TP/FN/FP diff` —
with dice/score in the title. Used by  `api.evaluate(save_overlays_dir=...)`
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from impact_team_2.visual.utils import resize_mask

PathLike = Union[str, Path]
ImageLike = Union[np.ndarray, "PILImage.Image"]  # noqa: F821


def _to_rgb_uint8(image: ImageLike) -> np.ndarray:
    from PIL import Image as PILImage
    if isinstance(image, PILImage.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr[..., :3]


def save_overlay(
    image: ImageLike,
    gt_mask: Optional[np.ndarray],
    pred_mask: Optional[np.ndarray],
    out_path: PathLike,
    *,
    dice: Optional[float] = None,
    score: Optional[float] = None,
    title: str = "",
) -> Path:
    """Write a 4-panel overlay PNG.

    Panels: image, image+GT (green), image+pred (magenta), diff (green=TP,
    red=FN, blue=FP). Masks that don't match the image resolution are
    nearest-neighbor resized to the image shape. `gt_mask` or `pred_mask`
    may be None — the corresponding panels render the image alone.

    Returns the path the file was written to.
    """
    import matplotlib.pyplot as plt

    img_rgb = _to_rgb_uint8(image)
    h, w = img_rgb.shape[:2]

    gt = resize_mask(gt_mask, (h, w)) if gt_mask is not None else None
    pr = resize_mask(pred_mask, (h, w)) if pred_mask is not None else None

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    header = title or ""
    if dice is not None:
        header = f"{header}  dice={dice:.4f}" if header else f"dice={dice:.4f}"
    if score is not None:
        header = f"{header}  score={score:.4f}"
    if header:
        fig.suptitle(header, fontsize=11)

    for ax in axes:
        ax.imshow(img_rgb)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_title("image")

    axes[1].set_title("GT")
    if gt is not None:
        overlay = np.zeros((h, w, 4))
        overlay[gt] = [0, 1, 0, 0.45]
        axes[1].imshow(overlay)

    axes[2].set_title("pred")
    if pr is not None:
        overlay = np.zeros((h, w, 4))
        overlay[pr] = [1, 0, 1, 0.45]
        axes[2].imshow(overlay)

    axes[3].set_title("diff (TP=green, FN=red, FP=blue)")
    if gt is not None and pr is not None:
        diff = np.zeros((h, w, 4))
        tp = gt & pr
        fn = gt & ~pr
        fp = ~gt & pr
        diff[tp] = [0, 1, 0, 0.5]
        diff[fn] = [1, 0, 0, 0.5]
        diff[fp] = [0, 0.4, 1, 0.5]
        axes[3].imshow(diff)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def resolve_save_indices(
    dice_scores: list[float],
    how: Union[int, str],
) -> list[int]:
    """Pick which val-set indices to save overlays for.

    `how` is one of:
      * int N           — first N indices (0..N-1)
      * "all"           — every index
      * "worst:K"       — K lowest-dice indices
      * "best:K"        — K highest-dice indices
    """
    n = len(dice_scores)
    if isinstance(how, int):
        return list(range(min(how, n)))
    if not isinstance(how, str):
        raise TypeError(f"how must be int or str, got {type(how).__name__}")
    if how == "all":
        return list(range(n))
    if ":" in how:
        mode, k_str = how.split(":", 1)
        k = int(k_str)
        if k <= 0:
            return []
        order = np.argsort(dice_scores)  # ascending
        if mode == "worst":
            return order[:k].tolist()
        if mode == "best":
            return order[-k:][::-1].tolist()
    raise ValueError(
        f"unknown overlay selection {how!r}; expected int, 'all', 'worst:K', or 'best:K'"
    )


__all__ = ["save_overlay", "resolve_save_indices"]
