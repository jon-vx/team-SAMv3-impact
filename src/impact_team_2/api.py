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
from typing import Callable, Literal, Optional, Sequence, overload

import numpy as np
from tqdm import tqdm

from impact_team_2.visual.utils import dice_score, resize_mask, summarize_dice


Model = str   # "SAM" | "MedSAM"
Mode = str    # "not_finetuned" | "finetuned"
PredictFn = Callable[..., dict]

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

_predictor_cache: dict[tuple[Model, Mode], PredictFn] = {}


def clear_cache() -> None:
    """Drop cached predictors and free their GPU memory.

    Call this between phases (e.g. after baseline eval, before training, and
    after training before finetuned eval) so the cached base models don't
    hold onto VRAM while the next phase builds fresh ones.
    """
    import gc
    import torch

    _predictor_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _medsam_finetuned_checkpoint() -> Path:
    return MEDSAM_FINETUNED_DIR / "best_lora_weights.pt"


def _sam_finetuned_checkpoint() -> Path:
    return SAM_FINETUNED_DIR / "sam3_finetuned_weights.safetensors"


def _build_predictor(model: Model, mode: Mode) -> PredictFn:
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


@overload
def predict(
    image,
    prompt: str,
    *,
    model: Model = ...,
    mode: Mode = ...,
    threshold: float = ...,
    return_details: Literal[False] = False,
) -> Optional[np.ndarray]: ...
@overload
def predict(
    image,
    prompt: str,
    *,
    model: Model = ...,
    mode: Mode = ...,
    threshold: float = ...,
    return_details: Literal[True],
) -> dict: ...
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

def evaluate(
    *,
    model_list: Sequence[Model],
    images: np.ndarray,
    ground_truth: np.ndarray,
    modes: Sequence[Mode] = ("not_finetuned",),
    prompt: str = "object",
    threshold: float = 0.01,
    save_overlays_dir: Optional[Path] = None,
    save_overlays_n: "int | str" = 0,
) -> dict:
    """Evaluate one or more (model, mode) combinations.

    The caller is responsible for passing held-out / external data. Returns
    a nested dict keyed by ``"<model>/<mode>"`` strings, each containing
    per-image dice scores and a summary.

    If ``save_overlays_dir`` is set, a 4-panel overlay PNG
    (image | GT | pred | diff) is written per selected val image. Selection
    is controlled by ``save_overlays_n``:

      * ``0``              — save nothing (default)
      * ``int N``          — save the first N val images
      * ``"all"``          — save every val image
      * ``"worst:K"``      — save the K lowest-dice images
      * ``"best:K"``       — save the K highest-dice images
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

    if save_overlays_dir is not None and save_overlays_n != 0:
        overlays_root: Optional[Path] = Path(save_overlays_dir)
    else:
        overlays_root = None

    results: dict[str, dict] = {}
    n = images.shape[0]

    for model in model_list:
        for mode in modes:
            pred_fn = _build_predictor(model, mode)
            key = f"{model}/{mode}"
            dice_scores: list[float] = []
            all_scores: list[float] = []
            per_image: dict[int, dict] = {}
            # Minimal per-image payload we need to render overlays at the end.
            # Keeping only `pred_mask`, `score`, and the PIL image from the
            # predictor output — the full `out` dict lives in `per_image`.
            overlay_buf: list[dict] = []

            for i in tqdm(range(n), desc=key):
                out = pred_fn(images[i], prompt, threshold=threshold)
                per_image[i] = out
                if out.get("scores") is not None:
                    all_scores.extend(out["scores"].tolist())

                pred_mask = _best_mask(out)
                if pred_mask is None:
                    dice_scores.append(0.0)
                    if overlays_root is not None:
                        overlay_buf.append({"pred": None, "score": None, "img": out.get("image")})
                    continue
                gt = resize_mask(ground_truth[i].astype(bool), pred_mask.shape)
                dice_scores.append(dice_score(pred_mask, gt))
                if overlays_root is not None:
                    scores_arr = out.get("scores")
                    best_score = (
                        float(scores_arr[int(scores_arr.argmax())])
                        if scores_arr is not None else None
                    )
                    overlay_buf.append({
                        "pred": pred_mask,
                        "score": best_score,
                        "img": out.get("image"),
                    })

            if overlays_root is not None:
                _write_overlays(
                    out_dir=overlays_root / key.replace("/", "_"),
                    tag=key,
                    images=images,
                    ground_truth=ground_truth,
                    dice_scores=dice_scores,
                    overlay_buf=overlay_buf,
                    how=save_overlays_n,
                )

            results[key] = {
                "results": per_image,
                "dice": dice_scores,
                "all_scores": all_scores,
                "summary": summarize_dice(dice_scores, all_scores),
            }

    return results


def _write_overlays(
    *,
    out_dir: Path,
    tag: str,
    images: np.ndarray,
    ground_truth: np.ndarray,
    dice_scores: list[float],
    overlay_buf: list[dict],
    how: "int | str",
) -> None:
    from impact_team_2.visual.overlays import save_overlay, resolve_save_indices

    indices = resolve_save_indices(dice_scores, how)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = tag.replace("/", "_")
    for i in indices:
        buf = overlay_buf[i]
        base_img = buf["img"] if buf["img"] is not None else images[i]
        parts = [f"{safe_tag}_val_{i:03d}", f"dice{dice_scores[i]:.3f}"]
        if buf["score"] is not None:
            parts.append(f"score{buf['score']:.3f}")
        fname = "_".join(parts) + ".png"
        save_overlay(
            base_img,
            ground_truth[i],
            buf["pred"],
            out_dir / fname,
            dice=dice_scores[i],
            score=buf["score"],
            title=f"{tag} · val[{i}]",
        )


__all__ = ["predict", "evaluate", "clear_cache"]
