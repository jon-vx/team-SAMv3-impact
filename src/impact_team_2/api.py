"""Top-level unified API for SAM3 / MedSAM3 inference and evaluation.

Usage:

    import impact_team_2 as I

    # Simple: returns the best predicted mask (bool ndarray).
    mask = I.predict(images[0], prompt="spleen", model="SAM")
    mask = I.predict(images[0], prompt="spleen", model="MedSAM", mode="finetuned")

    # Researcher: multi-model / multi-mode evaluation.
    results = I.evaluate(
        model_list=["SAM", "MedSAM"],
        images=images, ground_truth=masks,
        modes=["not_finetuned", "finetuned"],
    )

The caller is responsible for passing held-out / external data to
``evaluate``; nothing here checks for train/eval overlap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from tqdm import tqdm


Model = str   # "SAM" | "MedSAM"
Mode = str    # "not_finetuned" | "finetuned"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RUNS_DIR = _REPO_ROOT / "runs"

SAM_FINETUNED_DIR = _RUNS_DIR / "sam3_finetune"
MEDSAM_FINETUNED_DIR = _RUNS_DIR / "medsam3_finetune"
_UNET_WEIGHTS = _REPO_ROOT / "checkpoints" / "best_unetp.weights.h5"

_SUPPORTED_MODELS = ("SAM", "MedSAM")
_SUPPORTED_MODES = ("not_finetuned", "finetuned")


# ---------------------------------------------------------------------------
# Predictor resolution
# ---------------------------------------------------------------------------

_predictor_cache: dict[tuple[Model, Mode], object] = {}


def clear_cache() -> None:
    """Drop cached predictors and free their GPU memory.

    Call this between phases (e.g. after baseline eval, before training, and
    after training before finetuned eval) so the cached base models don't
    hold onto VRAM while the next phase builds fresh ones.
    """
    import gc
    _predictor_cache.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _medsam_finetuned_checkpoint() -> Path:
    return MEDSAM_FINETUNED_DIR / "best_lora_weights.pt"


def _sam_finetuned_checkpoint() -> Path:
    return SAM_FINETUNED_DIR / "sam3_finetuned_weights.safetensors"


def _build_predictor(model: Model, mode: Mode):
    key = (model, mode)
    if key in _predictor_cache:
        return _predictor_cache[key]

    if model == "MedSAM":
        from impact_team_2.inference._inference_medsam3 import build_predictor
        if mode == "not_finetuned":
            pred = build_predictor()  # public baseline
        else:
            ckpt = _medsam_finetuned_checkpoint()
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"MedSAM finetuned checkpoint not found at {ckpt}. "
                    f"Run `impact_team_2.train.train_medsam3` first."
                )
            pred = build_predictor(ckpt)
    elif model == "SAM":
        from impact_team_2.inference._inference_sam3 import build_predictor
        unet_weights = _UNET_WEIGHTS if _UNET_WEIGHTS.exists() else None
        if mode == "not_finetuned":
            pred = build_predictor(unet_weights=unet_weights)
        else:
            ckpt = _sam_finetuned_checkpoint()
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"SAM finetuned checkpoint not found at {ckpt}. "
                    f"Run `impact_team_2.train.sam.train_sam` first."
                )
            pred = build_predictor(weights_path=ckpt, unet_weights=unet_weights)
    else:
        raise ValueError(f"unknown model {model!r}; expected one of {_SUPPORTED_MODELS}")

    _predictor_cache[key] = pred
    return pred


# ---------------------------------------------------------------------------
# Public: predict
# ---------------------------------------------------------------------------

def _best_mask(result: dict) -> Optional[np.ndarray]:
    if result.get("masks") is None or result.get("scores") is None:
        return None
    return result["masks"][int(result["scores"].argmax())].astype(bool)


def predict(
    image,
    prompt: str,
    *,
    model: Model = "MedSAM",
    mode: Mode = "not_finetuned",
    threshold: float = 0.5,
    return_details: bool = False,
):
    """Run a single-image prediction.

    Returns the best predicted mask as a bool ndarray by default. Pass
    ``return_details=True`` to get the full backend dict (boxes, all scores,
    all masks, etc.).
    """
    if model not in _SUPPORTED_MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {_SUPPORTED_MODELS}")
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {_SUPPORTED_MODES}")

    pred_fn = _build_predictor(model, mode)
    result = pred_fn(image, prompt, threshold=threshold)
    if return_details:
        return result
    return _best_mask(result)


# ---------------------------------------------------------------------------
# Public: evaluate
# ---------------------------------------------------------------------------

def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 0.0
    return float(2 * inter / (denom + 1e-8))


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    from PIL import Image as PILImage
    return np.array(
        PILImage.fromarray(mask.astype(np.uint8)).resize(
            (shape[1], shape[0]), PILImage.NEAREST
        ),
        dtype=bool,
    )


def evaluate(
    *,
    model_list: Sequence[Model],
    images: np.ndarray,
    ground_truth: np.ndarray,
    modes: Sequence[Mode] = ("not_finetuned",),
    prompt: str = "object",
    threshold: float = 0.01,
) -> dict:
    """Evaluate one or more (model, mode) combinations.

    The caller is responsible for passing held-out / external data. Returns
    a nested dict keyed by ``"<model>/<mode>"`` strings, each containing
    per-image dice scores and a summary.
    """
    if images.shape[0] != ground_truth.shape[0]:
        raise ValueError("images and ground_truth must have the same length")

    for model in model_list:
        if model not in _SUPPORTED_MODELS:
            raise ValueError(
                f"unknown model {model!r}; expected one of {_SUPPORTED_MODELS}"
            )
    for mode in modes:
        if mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"unknown mode {mode!r}; expected one of {_SUPPORTED_MODES}"
            )

    results: dict[str, dict] = {}
    n = images.shape[0]

    for model in model_list:
        for mode in modes:
            pred_fn = _build_predictor(model, mode)
            key = f"{model}/{mode}"
            dice_scores: list[float] = []
            all_scores: list[float] = []
            per_image: dict[int, dict] = {}

            for i in tqdm(range(n), desc=key):
                out = pred_fn(images[i], prompt, threshold=threshold)
                per_image[i] = out
                if out.get("scores") is not None:
                    all_scores.extend(out["scores"].tolist())

                pred_mask = _best_mask(out)
                if pred_mask is None:
                    dice_scores.append(0.0)
                    continue
                gt = _resize_mask(ground_truth[i].astype(bool), pred_mask.shape)
                dice_scores.append(_dice(pred_mask, gt))

            results[key] = {
                "results": per_image,
                "dice": dice_scores,
                "all_scores": all_scores,
                "summary": _summarize(dice_scores, all_scores),
            }

    return results


def _summarize(
    dice_scores: Sequence[float], scores: Sequence[float]
) -> dict:
    out: dict = {
        "n": len(dice_scores),
        "mean_dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "max_dice": float(np.max(dice_scores)) if dice_scores else 0.0,
        "min_dice": float(np.min(dice_scores)) if dice_scores else 0.0,
        "dice_gt_0.5": int(sum(1 for d in dice_scores if d > 0.5)),
        "dice_gt_0.3": int(sum(1 for d in dice_scores if d > 0.3)),
    }
    if len(scores) > 0:
        out["score_min"] = float(np.min(scores))
        out["score_max"] = float(np.max(scores))
        out["score_mean"] = float(np.mean(scores))
    return out


__all__ = ["predict", "evaluate", "clear_cache"]
