# Spleen Segmentation with SAM3 and MedSAM3

Programmatic API for fine-tuning and running **SAM3** (Meta's HuggingFace
`facebook/sam3`) and **MedSAM3** (Liu et al.'s medical-domain LoRA fine-tune
on top of the native sam3 stack). Both models share a single unified API, and
fine-tuning + evaluation workflows for both are covered below.

## Setup


```bash
git clone https://github.com/jon-vx/team-SAMv3-impact.git
cd team-SAMv3-impact
source setup.sh # with conda in (base)
```

A Hugging Face token is required at runtime. `impact_team_2` auto-loads `.env.local`
on import, and `huggingface_hub` / `transformers` pick up `HF_TOKEN` from the
environment — so just drop the token into `.env.local`:

```bash
echo "HF_TOKEN=hf_xxxxxxxxxxxxx" > .env.local
```

## Model comparison

|                   | SAM3                                            | MedSAM3                                       |
|-------------------|-------------------------------------------------|-----------------------------------------------|
| Backbone          | `facebook/sam3` (HuggingFace Transformers)      | `lal-Joey/MedSAM3_v1` (native sam3 + LoRA)    |
| Domain            | General-purpose, vanilla weights                | LoRA-adapted on medical imagery               |
| Bounding box      | Optional — text-only by default; UNet (INIA) cascade available | Detected from the text prompt |
| Fine-tune style   | Unfreeze mask decoder                           | LoRA adapters on vision / text / mask decoder |

## Inference

Unified API:

```python
import impact_team_2 as I

# Default: MedSAM3, public weights.
mask = I.predict("image.png", prompt="spleen", model="MedSAM")

# SAM3 with a fine-tuned checkpoint.
mask = I.predict("image.png", prompt="spleen", model="SAM", mode="finetuned")
```

### MedSAM3

```python
from impact_team_2.inference import predict, build_predictor

# Lazy-built predictor backed by lal-Joey/MedSAM3_v1
result = predict("image.png", "spleen", threshold=0.5)
# result: {"boxes", "scores", "masks", "num_detections", "image"}

# Predictor bound to a specific LoRA checkpoint:
my_predictor = build_predictor("runs/medsam3_finetune/best_lora_weights.pt")
result = my_predictor(image_array, "spleen", threshold=0.5)
```

`image` can be a file path, a PIL Image, or a numpy array (HxW or HxWx3, uint8 or float).
`build_predictor(None)` downloads and uses the public MedSAM3 LoRA weights;
`build_predictor(path)` loads any LoRA `.pt` file with the same shape (e.g. the output
of `train_medsam3`).

The module-level `predict` builds its model lazily on first call, so importing the
module is cheap. Make sure `HF_TOKEN` is set in `.env.local` (or your shell) before
the first model download.

### SAM3

SAM3 uses a two-stage prompt: a vendor Keras UNet (INIA) predicts a segmentation, a bounding box is extracted and scaled back to the original image
dimensions, and SAM3 uses that box plus a text prompt to produce the final mask.

```python
from impact_team_2.inference._inference_sam3 import build_predictor, load_unet_weights

# Option 1 — load the UNet weights explicitly (cached across calls)
unet = load_unet_weights("checkpoints/best_unetp.weights.h5")
predictor = build_predictor(unet_model=unet)
result = predictor("scan.png", text_prompt="spleen")

# Option 2 — hand the weights path to build_predictor directly
predictor = build_predictor(unet_weights="checkpoints/best_unetp.weights.h5")

# Option 3 — layer a fine-tuned SAM3 checkpoint on top
predictor = build_predictor(
    unet_weights="checkpoints/best_unetp.weights.h5",
    weights_path="runs/sam3_finetune/sam3_finetuned_weights.safetensors",
)
```

**Output** (same shape as the MedSAM3 predictor):

```python
{
    "boxes":          np.ndarray,   # (N, 4) xyxy
    "scores":         np.ndarray,   # (N,)
    "masks":          np.ndarray,   # (N, H, W) bool
    "num_detections": int,
    "image":          PIL.Image,
}
```


| Parameter     | Default    | Description                                                     |
|---------------|------------|-----------------------------------------------------------------|
| `unet_weights`| `None`     | Path to `.h5` UNet weights — built and cached on first call     |
| `unet_model`  | `None`     | Pre-built Keras UNet model (alternative to `unet_weights`)      |
| `weights_path`| `None`     | Fine-tuned SAM3 `.safetensors` checkpoint                       |
| `text_prompt` | required   | Text description of the target region                           |
| `box`         | `None`     | Manual bbox `[x1, y1, x2, y2]` — overrides UNet bbox if present |
| `threshold`   | `0.5`      | Confidence threshold for both SAM3 and UNet                     |

## Training

### MedSAM3

```python
import numpy as np
from impact_team_2.train import train_medsam3

images = np.load("datasets/images.npz")["images"]      # (N, H, W, 3) uint8
masks  = np.load("datasets/masks.npz")["masks"] > 0    # (N, H, W) bool

best_weights = train_medsam3(
    images, masks,
    output_dir="runs/medsam3_finetune",
    category="spleen",
)
```

### SAM3

```python
import numpy as np
from impact_team_2.train.sam import train_sam

images = np.load("datasets/images.npz")["images"]      # (N, H, W, 3) uint8
masks  = np.load("datasets/masks.npz")["masks"] > 0    # (N, H, W) bool

ckpt = train_sam(
    images, masks,
    output_dir="runs/sam3_finetune",
    epochs=10,
    val_split=0.1,
    box_source="none",       # "none" (text-only, default), "unet", or "gt"
    text_prompt="spleen",    # used when box_source="none"
)
```

`box_source` controls what SAM3 sees as a prompt during training:

- `"none"` (default) — text-only. SAM3 learns from the text prompt alone; no UNet needed at inference.
- `"unet"` — boxes come from the UNet++ detector (detect-then-segment cascade). Pass a loaded Keras UNet via `unet_model=...`. UNet misses fall back to a full-image box so training stays UNet-free.
- `"gt"` — boxes derived from the ground-truth mask. Upper-bound baseline only; SAM3 won't get this free signal at inference.

Only the mask decoder is unfrozen, so the checkpoint at
`runs/sam3_finetune/sam3_finetuned_weights.safetensors` contains just the
trainable tensors (a few MB) and is loaded on top of the public HF SAM3 weights
via `load_state_dict(..., strict=False)`. Per-epoch metrics and loss curves are
written to `runs/sam3_finetune/tb/` (TensorBoard). The best-val-dice checkpoint
goes to the canonical filename; the most recent epoch lands in
`last_sam3_finetuned_weights.safetensors` as a backup.

## Visualization

<img width="1083" height="760" alt="image" src="https://github.com/user-attachments/assets/4dbd23a6-49ee-4542-8c14-fb64e1ad2dd3" />


```python
from impact_team_2.inference import build_predictor
from impact_team_2.visual import (
    evaluate, show_prediction_grid, show_training_curves,
    worst_dice, best_dice,
    save_overlay, save_comparison_overlay, save_contact_sheet,
    dice_score,
)

predictor = build_predictor("runs/medsam3_finetune/best_lora_weights.pt")
out = evaluate(predictor, images, masks, val_idx, prompt="spleen", threshold=0.5)
print(out["summary"])    # mean/max/min dice, dice>0.5, dice>0.3, score range

show_prediction_grid(images, masks, out["results"], title="Fine-tuned")
show_training_curves("runs/medsam3_finetune/val_stats.json")
show_prediction_grid(images, masks, worst_dice(out, k=5), title="Worst dice")

# Ad-hoc 4-panel overlay (image | GT | pred | diff) for a single case
save_overlay(images[0], masks[0], pred_mask, "runs/overlay.png",
             dice=dice_score(pred_mask, masks[0]), title="val[0]")

# Per-image baseline-vs-finetuned grid across multiple models
save_comparison_overlay(
    images[0], masks[0],
    baseline_preds={"SAM": sam_base_pred, "MedSAM": med_base_pred},
    finetuned_preds={"SAM": sam_ft_pred, "MedSAM": med_ft_pred},
    out_path="runs/compare/val0.png",
    baseline_dice={"SAM": 0.71, "MedSAM": 0.78},
    finetuned_dice={"SAM": 0.84, "MedSAM": 0.86},
)

# Dataset-wide contact sheet — one tile per val image for a single (model, mode)
save_contact_sheet(
    images, masks, [r["mask"] for r in out["results"]],
    out_path="runs/contact/medsam_finetuned.png",
    dice_scores=out["dice"], cols=7, title="MedSAM finetuned",
)
```

Three complementary views: `save_overlay` for one case, `save_comparison_overlay`
for cross-model sanity checks on a single image, and `save_contact_sheet` for
spotting failure patterns across the whole val split at a glance.
`examples/demo_api.py` wires all three into the end-to-end run.

## Examples

| Script                               | What it does                                                                                                            |
|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `examples/demo_api_predict.py`       | Single-image `I.predict` showcase: run both models on one image, write 4-panel overlays.                                |
| `examples/demo_medsam3_train.py`     | Download the public ultrasound spleen dataset and fine-tune MedSAM3 on it.                                              |
| `examples/demo_medsam3_eval.py`      | `I.evaluate` baseline vs fine-tuned MedSAM3, plus loss curves and worst-case grid.                                      |
| `examples/demo_sam3_train.py`        | Fine-tune SAM3 on the spleen dataset end-to-end. Writes to `runs/sam3_finetune/`.                                       |
| `examples/demo_sam3_unet_cascade.py` | SAM3 baseline (text-only) → train UNet++ bbox generator → SAM3 with UNet cascade, side-by-side summary.                 |
| `examples/demo_api.py`               | End-to-end unified-API demo: baseline eval → fine-tune SAM3 + MedSAM3 → eval → side-by-side comparison.                 |

```bash
conda activate impact-team-2

# Single-image predict (quickest smoke test)
python examples/demo_api_predict.py

# MedSAM3 flow
python examples/demo_medsam3_train.py    # train (writes runs/medsam3_finetune/)
python examples/demo_medsam3_eval.py     # plot before/after

# SAM3 flow
python examples/demo_sam3_unet_cascade.py                   # text-only → train UNet → SAM3+UNet cascade
python examples/demo_sam3_train.py                          # text-only, 10 epochs (default)
python examples/demo_sam3_train.py --box-source unet \
    --unet checkpoints/best_unetp.weights.h5                # detect-then-segment
python examples/demo_sam3_train.py --box-source gt          # GT-box upper bound
python examples/demo_sam3_train.py --n 32 --epochs 2        # quick sanity run

# Unified API end-to-end
python examples/demo_api.py
```


## Project Structure

```
.
├── checkpoints/                   # model weights (gitignored)
├── datasets/                      # training/eval data (gitignored)
├── runs/                          # training output: configs, COCO dataset, LoRA/SAM3 checkpoints, tb/
├── examples/                      # runnable example scripts
├── environment.yml                # conda env spec (Linux + cu126 + TF 2.19)
├── setup.sh                       # conda env create/update + CUDA-aware torch/TF install
└── src/impact_team_2/
    ├── api.py                     # unified predict / evaluate API
    ├── inference/
    │   ├── _inference_medsam3.py  # native sam3 + LoRA inference
    │   └── _inference_sam3.py     # HF Transformers SAM3 + UNet bbox
    ├── train/
    │   ├── med_sam.py             # train_medsam3 entry point
    │   └── sam.py                 # SAM3 trainer (mask-decoder only)
    ├── vendor/medsam3/            # vendored MedSAM3: lora_layers + train_sam3_lora_native
    ├── vendor/team_one/INIA/      # vendored UNet++ bbox generator
    └── visual/                    # evaluate, prediction grid + 4-panel overlays, training curves, pure mask helpers (dice / resize / summarize)
```

## Acknowledgments

- [SAM3](https://github.com/facebookresearch/sam3) — Meta AI
- [MedSAM3](https://github.com/Joey-S-Liu/MedSAM3) — Liu et al., 2025
- Spleen ultrasound dataset — [DivyanshuTak/Ultrasoud_Unet_Segmentation](https://github.com/DivyanshuTak/Ultrasoud_Unet_Segmentation)
