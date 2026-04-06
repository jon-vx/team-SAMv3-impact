import sam3
from pathlib import Path
from huggingface_hub import snapshot_download
from impact_team_2.vendor.medsam3.lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
from sam3.model_builder import build_sam3_image_model
from PIL import Image as PILImage
from sam3.train.data.sam3_image_dataset import Datapoint, Image as SAMImage, FindQueryLoaded, InferenceMetadata
from sam3.train.data.collator import collate_fn_api
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from torchvision.ops import nms
import torch.nn.functional as F
import numpy as np
import torch

print("test")

bpe_path = str(Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz")
lora_weights_path = snapshot_download(repo_id="lal-Joey/MedSAM3_v1")
lora_weights_file = next(Path(lora_weights_path).glob("*.pt"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_sam3_image_model(
    device=str(device),
    load_from_HF=True,
    bpe_path=bpe_path,
    eval_mode=True
)



lora_config = LoRAConfig(
    rank=16,
    alpha=32,
    dropout=0.0,
    target_modules=None,
    apply_to_vision_encoder=True,
    apply_to_text_encoder=True,
    apply_to_geometry_encoder=True,
    apply_to_detr_encoder=True,
    apply_to_detr_decoder=True,
    apply_to_mask_decoder=True,
)

model = apply_lora_to_model(model, lora_config)

load_lora_weights(model, str(lora_weights_file))
model.to(device)
model.eval()
print("Model ready")

RESOLUTION = 1008
transform = ComposeAPI(transforms=[
    RandomResizeAPI(sizes=RESOLUTION, max_size=RESOLUTION, square=True, consistent_transform=False),
    ToTensorAPI(),
    NormalizeAPI(mean=[0.5]*3, std=[0.5]*3),
])

def _create_datapoint(pil_image, text_prompt):
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

def _to_pil(image):
    if isinstance(image, str):
        return PILImage.open(image).convert("RGB")
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    img = np.stack([image]*3, axis=-1) if image.ndim == 2 else image
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return PILImage.fromarray(img, mode="RGB")

@torch.no_grad()
def predict(image, text_prompt, threshold=0.5, nms_iou=0.5):
    pil_image = _to_pil(image)
    orig_w, orig_h = pil_image.size

    batch = collate_fn_api([transform(_create_datapoint(pil_image, text_prompt))], dict_key="input")["input"]
    batch = copy_data_to_device(batch, device, non_blocking=True)
    out = model(batch)[-1]

    scores = out["pred_logits"].sigmoid()[0].max(dim=-1)[0]
    keep = scores > threshold
    if not keep.any():
        return {"boxes": None, "scores": None, "masks": None, "num_detections": 0, "image": pil_image}

    boxes_cxcywh = out["pred_boxes"][0, keep]
    kept_scores = scores[keep]

    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], device=cx.device)
    boxes_xyxy = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1) * scale

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
