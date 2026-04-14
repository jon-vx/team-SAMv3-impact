# SAM3 (HuggingFace Transformers) inference — vanilla, no LoRA.

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

PathLike = Union[str, Path]
PredictFn = Callable[..., dict]

_UNET_SIZE = 320   # input resolution expected by the vendor UNet (INIA)
_HF_MODEL_ID = "facebook/sam3"
_unet_cache: dict = {}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_pil(image) -> PILImage.Image:
    if isinstance(image, (str, Path)):
        return PILImage.open(image).convert("RGB")
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    arr = np.squeeze(image).astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255
    return PILImage.fromarray(arr.astype(np.uint8)).convert("RGB")


def _postprocess_mask(mask_array: np.ndarray) -> np.ndarray:
    """Morphological close + keep largest connected component."""
    m = mask_array.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    return (lbl == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(bool)


@torch.no_grad()
def _run_predict(model, processor, device, image, text_prompt, box, threshold) -> dict:
    pil_img = _to_pil(image)
    inputs = processor(
        images=pil_img,
        text=text_prompt,
        input_boxes=[[box]] if box is not None else None,
        return_tensors="pt",
    ).to(device)
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    if len(results["masks"]) == 0:
        return {"boxes": None, "scores": None, "masks": None, "num_detections": 0, "image": pil_img}

    raw_masks = results["masks"].cpu().numpy()
    raw_scores = results["scores"].cpu().numpy()
    processed_masks = np.stack([_postprocess_mask(raw_masks[i]) for i in range(len(raw_masks))])

    boxes_out = []
    for msk in processed_masks:
        ys, xs = np.where(msk)
        if len(xs) == 0:
            boxes_out.append([0, 0, pil_img.width, pil_img.height])
        else:
            boxes_out.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])

    return {
        "boxes": np.array(boxes_out, dtype=np.float32),
        "scores": raw_scores,
        "masks": processed_masks,
        "num_detections": len(processed_masks),
        "image": pil_img,
    }


def load_unet_weights(weights_path: PathLike):
    """Build UNet++ from INIA and load weights. Cached by path."""
    weights_path = str(weights_path)
    if weights_path in _unet_cache:
        return _unet_cache[weights_path]
    from impact_team_2.vendor.team_one.INIA import get_model
    unet = get_model("unet++")
    unet.predict(np.zeros((1, _UNET_SIZE, _UNET_SIZE, 1), dtype=np.float32), verbose=0)
    unet.load_weights(weights_path)
    print(f"[UNet] Loaded weights from {weights_path}")
    _unet_cache[weights_path] = unet
    return unet


def _box_from_unet(unet_model, pil_img: PILImage.Image, threshold: float = 0.5) -> Optional[list]:
    """Run INIA get_bboxes on the image and scale the result to original dimensions."""
    from impact_team_2.vendor.team_one.INIA import get_bboxes
    orig_w, orig_h = pil_img.size
    gray = np.array(pil_img.convert("L").resize((_UNET_SIZE, _UNET_SIZE))) / 255.0
    inp = gray[np.newaxis, :, :, np.newaxis].astype(np.float32)
    bbox = get_bboxes(unet_model, inp, threshold=threshold)[0]
    if bbox is None:
        return None
    sx, sy = orig_w / _UNET_SIZE, orig_h / _UNET_SIZE
    x_min, y_min, x_max, y_max = bbox
    return [int(x_min * sx), int(y_min * sy), int(x_max * sx), int(y_max * sy)]


def build_predictor(
    unet_model=None,
    unet_weights: Optional[PathLike] = None,
    weights_path: Optional[PathLike] = None,
) -> PredictFn:
    """Build a SAM3 predictor.

    Pass ``unet_weights`` or ``unet_model`` for auto bbox generation.
    Pass ``weights_path`` (a ``.safetensors`` file) to load fine-tuned
    SAM3 weights on top of the base HF model.
    """
    from transformers import Sam3Model, Sam3Processor
    device = _get_device()
    print(f"[SAM3] Loading model on device: {device}")
    model = Sam3Model.from_pretrained(_HF_MODEL_ID).to(device)
    processor = Sam3Processor.from_pretrained(_HF_MODEL_ID)

    if weights_path is not None:
        from safetensors.torch import load_file
        state = load_file(str(weights_path))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[SAM3] Loaded fine-tuned weights from {weights_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    model.eval()
    print(f"[SAM3] Model ready ({_HF_MODEL_ID})")

    _unet = load_unet_weights(unet_weights) if unet_weights is not None else unet_model

    def predict(image, text_prompt: str, box: Optional[list] = None, threshold: float = 0.5) -> dict:
        pil_img = _to_pil(image)
        if box is None and _unet is not None:
            box = _box_from_unet(_unet, pil_img, threshold=threshold)
        return _run_predict(model, processor, device, pil_img, text_prompt, box, threshold)

    return predict


_default_predictor: Optional[PredictFn] = None


def predict(image, text_prompt: str, box: Optional[list] = None, threshold: float = 0.5,
            unet_weights: Optional[PathLike] = None) -> dict:
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = build_predictor(unet_weights=unet_weights)
    return _default_predictor(image, text_prompt, box=box, threshold=threshold)
