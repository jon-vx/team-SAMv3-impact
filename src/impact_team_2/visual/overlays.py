"""Prediction overlay helpers.

Three complementary views for inspecting segmentation results:

  * `save_overlay` — per-image 4-panel (image | GT | pred | diff) for one
    `(model, mode)` combination.
  * `save_comparison_overlay` — per-image grid comparing baseline vs finetuned
    across multiple models on the same val image.
  * `save_contact_sheet` — dataset-wide grid showing one model's prediction
    on every val image, for at-a-glance failure-pattern spotting.
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


def save_comparison_overlay(
    image: ImageLike,
    gt_mask: Optional[np.ndarray],
    baseline_preds: dict[str, Optional[np.ndarray]],
    finetuned_preds: dict[str, Optional[np.ndarray]],
    out_path: PathLike,
    *,
    baseline_dice: Optional[dict[str, float]] = None,
    finetuned_dice: Optional[dict[str, float]] = None,
    title: str = "",
) -> Path:
    """Render a 2×(N+1) cross-model comparison for one val image.

    Row 1: image       | <baseline_preds[m0]>   | <baseline_preds[m1]>  | ...
    Row 2: GT overlay  | <finetuned_preds[m0]>  | <finetuned_preds[m1]> | ...

    `baseline_preds` and `finetuned_preds` must have the same keys (model
    names). Panel titles include the per-pred dice score when provided.
    Pred masks in magenta; GT in green.
    """
    import matplotlib.pyplot as plt

    if set(baseline_preds) != set(finetuned_preds):
        raise ValueError(
            f"baseline/finetuned preds must share keys; "
            f"baseline={sorted(baseline_preds)} finetuned={sorted(finetuned_preds)}"
        )
    models = list(baseline_preds)

    img_rgb = _to_rgb_uint8(image)
    h, w = img_rgb.shape[:2]
    gt = resize_mask(gt_mask, (h, w)) if gt_mask is not None else None

    n_cols = 1 + len(models)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))
    legend = "GT = green · pred = magenta"
    header = f"{title}\n{legend}" if title else legend
    fig.suptitle(header, fontsize=11)

    for ax in axes.flat:
        ax.imshow(img_rgb)
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 0, col 0: plain image
    axes[0, 0].set_title("image")

    # Row 1, col 0: GT overlay
    axes[1, 0].set_title("GT")
    if gt is not None:
        gt_overlay = np.zeros((h, w, 4))
        gt_overlay[gt] = [0, 1, 0, 0.45]
        axes[1, 0].imshow(gt_overlay)

    def _render_pred(ax, pred_mask, model_name, row_label, dice):
        pr = resize_mask(pred_mask, (h, w)) if pred_mask is not None else None
        t = f"{model_name} {row_label}"
        if dice is not None:
            t = f"{t}\ndice={dice:.3f}"
        ax.set_title(t, fontsize=9)
        if pr is not None:
            pr_overlay = np.zeros((h, w, 4))
            pr_overlay[pr] = [1, 0, 1, 0.45]
            ax.imshow(pr_overlay)

    for j, model in enumerate(models, start=1):
        _render_pred(
            axes[0, j], baseline_preds[model], model, "baseline",
            (baseline_dice or {}).get(model),
        )
        _render_pred(
            axes[1, j], finetuned_preds[model], model, "finetuned",
            (finetuned_dice or {}).get(model),
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_contact_sheet(
    images: list,
    gt_masks: list,
    pred_masks: list,
    out_path: PathLike,
    *,
    dice_scores: Optional[list] = None,
    cols: int = 7,
    title: str = "",
) -> Path:
    """Render a grid of tiles — one tile per val image — for a single
    (model, mode) combination.

    Each tile: image with GT contour (lime, thin) and pred fill (magenta,
    translucent). Tile subtitle is the dice score when provided.

    `images`, `gt_masks`, and `pred_masks` must be the same length; entries
    in `pred_masks` may be None (panels render image + GT only).
    """
    import matplotlib.pyplot as plt

    n = len(images)
    if len(gt_masks) != n or len(pred_masks) != n:
        raise ValueError(
            f"images / gt_masks / pred_masks length mismatch: "
            f"{n} vs {len(gt_masks)} vs {len(pred_masks)}"
        )
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    legend = "GT contour = green · pred fill = magenta"
    header = f"{title}\n{legend}" if title else legend
    fig.suptitle(header, fontsize=12)

    axes_flat = axes.flat if rows * cols > 1 else [axes]
    for i, ax in enumerate(axes_flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")
            continue
        img_rgb = _to_rgb_uint8(images[i])
        h, w = img_rgb.shape[:2]
        ax.imshow(img_rgb)

        if pred_masks[i] is not None:
            pr = resize_mask(pred_masks[i], (h, w))
            pr_overlay = np.zeros((h, w, 4))
            pr_overlay[pr] = [1, 0, 1, 0.40]
            ax.imshow(pr_overlay)

        if gt_masks[i] is not None:
            gt = resize_mask(gt_masks[i], (h, w))
            ax.contour(gt.astype(float), levels=[0.5], colors=["#66ff66"], linewidths=0.8)

        if dice_scores is not None and i < len(dice_scores):
            ax.set_title(f"[{i}] dice={dice_scores[i]:.3f}", fontsize=8)
        else:
            ax.set_title(f"[{i}]", fontsize=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


__all__ = [
    "save_overlay",
    "save_comparison_overlay",
    "save_contact_sheet",
    "resolve_save_indices",
]
