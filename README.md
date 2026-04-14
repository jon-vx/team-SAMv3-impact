# Spleen Segmentation with SAM3 and MedSAM3

Programmatic API for fine-tuning and running **SAM3** (Meta's HuggingFace
`facebook/sam3`) and **MedSAM3** (Liu et al.'s medical-domain LoRA fine-tune
on top of the native sam3 stack). Both models share a single unified API, and
fine-tuning + evaluation workflows for both are covered below.

## Setup

```bash
git clone https://github.com/jon-vx/team-SAMv3-impact.git
cd team-SAMv3-impact
./setup.sh        # or: source setup.sh   to also activate the venv in this shell
```

`setup.sh` creates `venv/`, detects your CUDA driver via `nvidia-smi` and installs the
matching PyTorch wheel (cu118 / cu121 / cu124 / cu126 / cu128, falling back to the CPU
wheel if no GPU), installs the project in editable mode, and reports whether CUDA is
visible to torch after the install.

A Hugging Face token is required at runtime. `impact_team_2` auto-loads `.env.local`
on import, and `huggingface_hub` / `transformers` pick up `HF_TOKEN` from the
environment — so just drop the token into `.env.local`:

```bash
echo "HF_TOKEN=hf_xxxxxxxxxxxxx" > .env.local
```

SAM3's automatic-bbox mode uses a vendor Keras UNet (INIA). That path isn't part of
the default install — opt in via the `unet` extra:

```bash
pip install -e ".[unet]"
```

On Apple Silicon, add `pip install tensorflow-metal` afterwards for Metal-accelerated
TF (Mac-only; installing it on Linux/Windows will break).

## Model comparison

|                   | SAM3                                            | MedSAM3                                       |
|-------------------|-------------------------------------------------|-----------------------------------------------|
| Backbone          | `facebook/sam3` (HuggingFace Transformers)      | `lal-Joey/MedSAM3_v1` (native sam3 + LoRA)    |
| Domain            | General-purpose, vanilla weights                | LoRA-adapted on medical imagery               |
| Bounding box      | Auto-generated via vendor UNet (INIA)           | Not required — detected from the text prompt  |
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
result = predictor("scan.png", text_prompt="spleen organ in ultrasound")

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
)
```

Only the mask decoder is unfrozen, so the checkpoint at
`runs/sam3_finetune/sam3_finetuned_weights.safetensors` contains just the
trainable tensors (a few MB) and is loaded on top of the public HF SAM3 weights
via `load_state_dict(..., strict=False)`. Per-epoch metrics and loss curves are
written to `runs/sam3_finetune/tb/` (TensorBoard). The best-val-dice checkpoint
goes to the canonical filename; the most recent epoch lands in
`last_sam3_finetuned_weights.safetensors` as a backup.

## Visualization

```python
from impact_team_2.inference import build_predictor
from impact_team_2.visual import (
    evaluate, show_prediction_grid, show_training_curves,
    worst_dice, best_dice,
)

predictor = build_predictor("runs/medsam3_finetune/best_lora_weights.pt")
out = evaluate(predictor, images, masks, val_idx, prompt="spleen", threshold=0.01)
print(out["summary"])    # mean/max/min dice, dice>0.5, dice>0.3, score range

show_prediction_grid(images, masks, out["results"], title="Fine-tuned")
show_training_curves("runs/medsam3_finetune/val_stats.json")
show_prediction_grid(images, masks, worst_dice(out, k=5), title="Worst dice")
```

## Examples

| Script                        | What it does                                                                                                            |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `examples/test_medsam3_inf.py`   | Run baseline MedSAM3 over the spleen val split, plot the prediction grid.                                            |
| `examples/test_medsam3_train.py` | Download the public ultrasound spleen dataset and fine-tune MedSAM3 on it.                                           |
| `examples/test_medsam3_eval.py`  | Full before/after loop for MedSAM3: baseline vs fine-tuned, plus loss curves and worst-case grid.                    |
| `examples/test_sam3_train.py`    | Fine-tune **SAM3** on the spleen dataset end-to-end. Writes to `runs/sam3_finetune/`.                                |
| `examples/test_sam3_inf.py`      | Evaluate SAM3 on the spleen val split (baseline or fine-tuned), report per-image dice + summary, save overlays.      |
| `examples/test_api.py`           | End-to-end unified-API demo: baseline eval → fine-tune SAM3 + MedSAM3 → eval → side-by-side comparison.              |

```bash
# MedSAM3 flow
venv/bin/python examples/test_medsam3_train.py    # train (writes runs/medsam3_finetune/)
venv/bin/python examples/test_medsam3_eval.py     # plot before/after

# SAM3 flow
venv/bin/python examples/test_sam3_train.py                 # full dataset, 10 epochs
venv/bin/python examples/test_sam3_train.py --epochs 20     # longer run
venv/bin/python examples/test_sam3_train.py --n 32 --epochs 2   # quick sanity run

venv/bin/python examples/test_sam3_inf.py                   # baseline eval on val split
venv/bin/python examples/test_sam3_inf.py --weights runs/sam3_finetune/sam3_finetuned_weights.safetensors
venv/bin/python examples/test_sam3_inf.py --train           # retrain UNet bbox generator first
```

`test_medsam3_train.py` lazy-downloads `images.npz` / `masks.npz` from
[DivyanshuTak/Ultrasoud_Unet_Segmentation](https://github.com/DivyanshuTak/Ultrasoud_Unet_Segmentation)
into `datasets/ultrasound_spleen/` on first run; the SAM3 scripts reuse the
same cache.

`test_sam3_inf.py` always evaluates against the held-out val split (seed `42`,
`val_split=0.1` to match training) and reports per-image dice plus a summary
(mean, median, min/max, `dice>0.5`, `dice>0.3`). Pass `--weights <path>` for a
fine-tuned checkpoint; omit it to benchmark the baseline. Overlays for the
first `--n-overlays` val images are written to `runs/` so baseline and
fine-tuned results can be compared visually. Requires the UNet++ bbox generator
at `checkpoints/best_unetp.weights.h5` — pass `--train` once to create it, or
`--unet <path>` to point at an existing one.

## Project Structure

```
.
├── checkpoints/                   # model weights (gitignored)
├── datasets/                      # training/eval data (gitignored)
├── runs/                          # training output: configs, COCO dataset, LoRA/SAM3 checkpoints, tb/
├── examples/                      # runnable example scripts
├── setup.sh                       # venv + CUDA-aware torch install
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
    └── visual/                    # evaluate, prediction grid, training curves, dice helpers
```

## Acknowledgments

- [SAM3](https://github.com/facebookresearch/sam3) — Meta AI
- [MedSAM3](https://github.com/Joey-S-Liu/MedSAM3) — Liu et al., 2025
- Spleen ultrasound dataset — [DivyanshuTak/Ultrasoud_Unet_Segmentation](https://github.com/DivyanshuTak/Ultrasoud_Unet_Segmentation)
