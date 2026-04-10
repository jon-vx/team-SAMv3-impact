"""MedSAM3 LoRA inference."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import sam3
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image as PILImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)
from torchvision.ops import nms

from impact_team_2.vendor.medsam3.lora_layers import (
    LoRAConfig,
    apply_lora_to_model,
    load_lora_weights,
)

_BPE_PATH = Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
if not _BPE_PATH.exists():
    import urllib.request
    _BPE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading missing BPE vocab -> {_BPE_PATH}")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/facebookresearch/sam3/main/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        _BPE_PATH,
    )
_BPE_PATH = str(_BPE_PATH)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_RESOLUTION = 1008

_TRANSFORM = ComposeAPI(transforms=[
    RandomResizeAPI(sizes=_RESOLUTION, max_size=_RESOLUTION, square=True, consistent_transform=False),
    ToTensorAPI(),
    NormalizeAPI(mean=[0.5] * 3, std=[0.5] * 3),
])

PathLike = Union[str, Path]
PredictFn = Callable[..., dict]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _create_datapoint(pil_image: PILImage.Image, text_prompt: str) -> Datapoint:
    w, h = pil_image.size
    return Datapoint(
        find_queries=[FindQueryLoaded(
            query_text=text_prompt, image_id=0, object_ids_output=[], is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=0, original_image_id=0, original_category_id=1,
                original_size=(w, h), object_id=0, frame_index=0,
            ),
        )],
        images=[SAMImage(data=pil_image, objects=[], size=(h, w))],
    )


def _to_pil(image) -> PILImage.Image:
    if isinstance(image, str):
        return PILImage.open(image).convert("RGB")
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    img = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return PILImage.fromarray(img, mode="RGB")


@torch.no_grad()
def _run_predict(model, image, text_prompt: str, threshold: float, nms_iou: float) -> dict:
    pil_image = _to_pil(image)
    orig_w, orig_h = pil_image.size

    batch = collate_fn_api(
        [_TRANSFORM(_create_datapoint(pil_image, text_prompt))], dict_key="input"
    )["input"]
    batch = copy_data_to_device(batch, _DEVICE, non_blocking=True)
    out = model(batch)[-1]

    scores = out["pred_logits"].sigmoid()[0].max(dim=-1)[0]
    keep = scores > threshold
    if not keep.any():
        return {"boxes": None, "scores": None, "masks": None,
                "num_detections": 0, "image": pil_image}

    boxes_cxcywh = out["pred_boxes"][0, keep]
    kept_scores = scores[keep]

    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], device=cx.device)
    boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1) * scale

    nms_keep = nms(boxes_xyxy, kept_scores, nms_iou)
    boxes_xyxy, kept_scores = boxes_xyxy[nms_keep], kept_scores[nms_keep]

    masks_np = None
    if (pred_masks := out.get("pred_masks")) is not None:
        masks_np = (F.interpolate(
            pred_masks[0, keep][nms_keep].sigmoid().unsqueeze(0).float(),
            size=(orig_h, orig_w), mode="bilinear", align_corners=False,
        ).squeeze(0) > 0.5).cpu().numpy()

    return {
        "boxes": boxes_xyxy.cpu().numpy(),
        "scores": kept_scores.cpu().numpy(),
        "masks": masks_np,
        "num_detections": len(nms_keep),
        "image": pil_image,
    }


# ---------------------------------------------------------------------------
# Predictor factory
# ---------------------------------------------------------------------------

_DEFAULT_LORA_REPO = "lal-Joey/MedSAM3_v1"


def _resolve_lora_path(lora_weights_path: Optional[PathLike]) -> Path:
    if lora_weights_path is not None:
        path = Path(lora_weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"LoRA weights not found at {path}")
        return path
    snapshot_dir = Path(snapshot_download(repo_id=_DEFAULT_LORA_REPO))
    weights = next(snapshot_dir.glob("*.pt"), None)
    if weights is None:
        raise FileNotFoundError(
            f"No .pt file found in MedSAM3 LoRA snapshot at {snapshot_dir}"
        )
    return weights


def build_predictor(lora_weights_path: Optional[PathLike] = None) -> PredictFn:
    """Build a `predict(image, text_prompt, ...)` callable bound to its own model.

    Args:
        lora_weights_path: path to a `*.pt` LoRA checkpoint (e.g. the output of
            `train_medsam3`). If `None`, downloads and uses the public
            `lal-Joey/MedSAM3_v1` weights.

    The returned callable has the signature:
        predict(image, text_prompt, threshold=0.5, nms_iou=0.5) -> dict
    where `image` is a path / PIL Image / numpy array (HxW or HxWx3), and the
    result dict matches the legacy `predict()` shape.
    """
    weights_path = _resolve_lora_path(lora_weights_path)

    model = build_sam3_image_model(
        device=str(_DEVICE),
        load_from_HF=True,
        bpe_path=_BPE_PATH,
        eval_mode=True,
    )
    model = apply_lora_to_model(
        model,
        LoRAConfig(
            rank=16, alpha=32, dropout=0.0, target_modules=None,
            apply_to_vision_encoder=True, apply_to_text_encoder=True,
            apply_to_geometry_encoder=True, apply_to_detr_encoder=True,
            apply_to_detr_decoder=True, apply_to_mask_decoder=True,
        ),
    )
    load_lora_weights(model, str(weights_path))
    model.to(_DEVICE)
    model.eval()
    print(f"MedSAM3 predictor ready (weights: {weights_path})")

    def predict(image, text_prompt: str, threshold: float = 0.5, nms_iou: float = 0.5) -> dict:
        return _run_predict(model, image, text_prompt, threshold, nms_iou)

    return predict


# ---------------------------------------------------------------------------
# Backward-compatible module-level `predict` (lazy default predictor)
# ---------------------------------------------------------------------------

_default_predictor: Optional[PredictFn] = None


def predict(image, text_prompt: str, threshold: float = 0.5, nms_iou: float = 0.5) -> dict:
    """Lazy default predictor backed by the public `lal-Joey/MedSAM3_v1` weights.

    The model is built on the first call. For multiple/custom LoRA checkpoints,
    use `build_predictor(...)` directly so each predictor owns its own model.
    """
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = build_predictor()
    return _default_predictor(image, text_prompt, threshold=threshold, nms_iou=nms_iou)
