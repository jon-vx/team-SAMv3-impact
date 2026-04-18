from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image as PILImage
from tqdm import tqdm

from impact_team_2.visual.utils import best_mask, dice_score, resize_mask, summarize_dice


PredictFn = Callable[..., dict]


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def plot_prediction_grid(
    images: np.ndarray,
    masks: np.ndarray,
    results: Mapping[int, dict],
    *,
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
) -> tuple[Figure, list[float]]:
    """Build the input | GT | prediction grid for each result, returning the figure.

    Args:
        images: full image stack (N, H, W, 3).
        masks: full GT mask stack (N, H, W).
        results: mapping from image index -> dict returned by `predict()`.
        title: optional suptitle (e.g. "Before fine-tuning").
        save_path: if given, the figure is also saved to this path.

    Returns:
        (figure, dice_scores) — `dice_scores` is one entry per result, in the
        order the keys of `results` are iterated.
    """
    indices = list(results.keys())
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n), squeeze=False)

    dice_scores: list[float] = []

    for row, idx in enumerate(indices):
        result = results[idx]
        gt = masks[idx].astype(bool)

        axes[row, 0].imshow(images[idx])
        axes[row, 0].set_title(f"Image {idx}", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(images[idx])
        gt_overlay = np.zeros((*gt.shape, 4))
        gt_overlay[gt] = [0, 1, 0, 0.4]
        axes[row, 1].imshow(gt_overlay)
        axes[row, 1].set_title("Ground Truth", fontsize=10)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(result["image"])
        pred_mask = best_mask(result)
        if pred_mask is not None:
            gt_for_dice = resize_mask(gt, pred_mask.shape)
            d = dice_score(pred_mask, gt_for_dice)
            dice_scores.append(d)

            pred_overlay = np.zeros((*pred_mask.shape, 4))
            pred_overlay[pred_mask] = [1, 0, 0, 0.4]
            axes[row, 2].imshow(pred_overlay)
            top_score = float(result["scores"].max())
            axes[row, 2].set_title(
                f"Pred — score: {top_score:.3f}, dice: {d:.3f}", fontsize=10
            )
        else:
            dice_scores.append(0.0)
            axes[row, 2].set_title("No detections (dice: 0.000)", fontsize=10)
        axes[row, 2].axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=14, y=1.0)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    return fig, dice_scores


def _is_interactive_backend() -> bool:
    """True if matplotlib's current backend can actually display a window."""
    backend = matplotlib.get_backend().lower()
    # Non-GUI backends never produce a window. The common headless one is "agg";
    # the cairo / pdf / svg / ps backends are also file-only.
    return backend not in {"agg", "cairo", "pdf", "svg", "ps", "template"}


def _show_or_save(fig: Figure, save_path: Optional[Path | str], fallback_name: str) -> None:
    """Display `fig` if the backend is interactive; otherwise save it to disk.

    Used by the `show_*` helpers so they all behave the same on headless boxes:
    a no-op `plt.show()` is replaced by a save to `save_path` (or a tempdir
    fallback if no path was given), and the path is printed.
    """
    if _is_interactive_backend():
        plt.show()
        return

    if save_path is None:
        save_path = Path(tempfile.gettempdir()) / fallback_name
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    print(
        f"[visual] non-interactive matplotlib backend ({matplotlib.get_backend()}) "
        f"— figure saved to {save_path}"
    )


def show_prediction_grid(
    images: np.ndarray,
    masks: np.ndarray,
    results: Mapping[int, dict],
    *,
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
) -> list[float]:
    """Build the prediction grid and display it.

    On an interactive matplotlib backend (local desktop, Jupyter, VS Code plot
    viewer) this opens a window. On a headless backend (e.g. SSH without X11)
    the figure is saved to `save_path` (or a temp PNG if not given) and the
    path is printed.

    Returns the per-image dice scores.
    """
    fig, dice_scores = plot_prediction_grid(
        images, masks, results, title=title, save_path=save_path
    )
    _show_or_save(fig, save_path, "medsam3_prediction_grid.png")
    return dice_scores


# ---------------------------------------------------------------------------
# Training curves (reads val_stats.json written by SAM3TrainerNative)
# ---------------------------------------------------------------------------

def _read_training_stats(stats_path: Path | str) -> list[dict]:
    """Parse the JSONL file the MedSAM3 trainer appends to each epoch."""
    rows: list[dict] = []
    with open(stats_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_training_curves(
    stats_path: Path | str,
    *,
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Plot train/val loss vs epoch from a `val_stats.json` file.

    `SAM3TrainerNative` appends one JSON line per epoch to
    `<output_dir>/val_stats.json` with `{epoch, train_loss, val_loss}`. This
    helper reads that file and renders both curves on a shared axis.

    Returns the figure (so the caller can save / show / further customize).
    """
    rows = _read_training_stats(stats_path)
    if not rows:
        raise ValueError(f"No training stats found in {stats_path}")

    epochs = [r["epoch"] for r in rows]
    train_losses = [r.get("train_loss") for r in rows]
    val_losses = [r.get("val_loss") for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    if any(v is not None for v in train_losses):
        ax.plot(epochs, train_losses, marker="o", label="train")
    if any(v is not None for v in val_losses):
        ax.plot(epochs, val_losses, marker="o", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title or "MedSAM3 LoRA training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    return fig


def show_training_curves(
    stats_path: Path | str,
    *,
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
) -> None:
    """Build the training-curves figure and display it (with headless fallback)."""
    fig = plot_training_curves(stats_path, title=title, save_path=save_path)
    _show_or_save(fig, save_path, "medsam3_training_curves.png")


# ---------------------------------------------------------------------------
# Convenience: run predictions + collect everything
# ---------------------------------------------------------------------------

def evaluate(
    predict_fn: PredictFn,
    images: np.ndarray,
    masks: np.ndarray,
    indices: Iterable[int],
    *,
    prompt: str,
    threshold: float = 0.01,
    desc: str = "evaluate",
) -> dict:
    indices = list(indices)
    results: dict[int, dict] = {}
    all_scores: list[float] = []

    for idx in tqdm(indices, desc=desc):
        result = predict_fn(images[idx], prompt, threshold=threshold)
        results[idx] = result
        if result.get("scores") is not None:
            all_scores.extend(result["scores"].tolist())

    # Reuse plot_prediction_grid's dice computation by inlining the cheap part
    # (we don't want to actually build a figure here).
    dice_scores: list[float] = []
    for idx in indices:
        result = results[idx]
        pred_mask = best_mask(result)
        if pred_mask is None:
            dice_scores.append(0.0)
            continue
        gt = resize_mask(masks[idx].astype(bool), pred_mask.shape)
        dice_scores.append(dice_score(pred_mask, gt))

    return {
        "results": results,
        "dice": dice_scores,
        "all_scores": all_scores,
        "summary": summarize_dice(dice_scores, all_scores),
    }


# ---------------------------------------------------------------------------
# Failure-case helpers
# ---------------------------------------------------------------------------

def _rank_by_dice(eval_out: dict, k: int, *, worst: bool) -> dict:
    """Internal: return the K results with the worst (or best) dice scores."""
    indices = list(eval_out["results"].keys())
    dice = eval_out["dice"]
    pairs = sorted(zip(indices, dice), key=lambda p: p[1], reverse=not worst)
    picked = pairs[:k]
    return {idx: eval_out["results"][idx] for idx, _ in picked}


def worst_dice(eval_out: dict, k: int = 5) -> dict:
    """Return the K result entries with the lowest dice score, in worst-first order.

    The return shape matches `evaluate(...)["results"]` so it can be passed
    straight to `plot_prediction_grid` / `show_prediction_grid` to visualize
    failure cases.
    """
    return _rank_by_dice(eval_out, k, worst=True)


def best_dice(eval_out: dict, k: int = 5) -> dict:
    """Return the K result entries with the highest dice score, in best-first order."""
    return _rank_by_dice(eval_out, k, worst=False)
