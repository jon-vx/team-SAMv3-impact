# Spleen Segmentation with SAM3 and MedSAM3

Programmatic API for fine-tuning and running SAM3 and MedSAM3

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

A Hugging Face token is required at runtime (the inference and training paths both call
`huggingface_hub.login(...)` at import time):

```bash
echo "HF_TOKEN=hf_xxxxxxxxxxxxx" > .env.local
```

## Inference

```python
from impact_team_2.inference import predict, build_predictor

# Default: lazy-built predictor backed by lal-Joey/MedSAM3_v1
result = predict("image.png", "spleen", threshold=0.5)
# result: {"boxes", "scores", "masks", "num_detections", "image"}

# Or build a predictor bound to a specific LoRA checkpoint:
my_predictor = build_predictor("runs/medsam3_finetune/best_lora_weights.pt")
result = my_predictor(image_array, "spleen", threshold=0.5)
```

`image` can be a file path, a PIL Image, or a numpy array (HxW or HxWx3, uint8 or float).
`build_predictor(None)` downloads and uses the public MedSAM3 LoRA weights;
`build_predictor(path)` loads any LoRA `.pt` file with the same shape (e.g. the output of
`train_medsam3`).

The module-level `predict` builds its model lazily on first call, so importing the
module is cheap. Importing `impact_team_2.inference` does call `huggingface_hub.login`
at import time, so make sure `HF_TOKEN` is set first.

## Training

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

| Script | What it does |
|---|---|
| `examples/test_inf.py` | Run baseline MedSAM3 over the spleen val split, plot the prediction grid. |
| `examples/test_train.py` | Download the public ultrasound spleen dataset and fine-tune MedSAM3 on it. |
| `examples/test_eval.py` | Full before/after loop: baseline vs fine-tuned, plus loss curves and worst-case grid. |

```bash
venv/bin/python examples/test_train.py    # train (writes runs/medsam3_finetune/)
venv/bin/python examples/test_eval.py     # plot before/after
```

`examples/test_train.py` lazy-downloads `images.npz` / `masks.npz` from
[DivyanshuTak/Ultrasoud_Unet_Segmentation](https://github.com/DivyanshuTak/Ultrasoud_Unet_Segmentation)
into `datasets/ultrasound_spleen/` on first run.

## Project Structure

```
.
├── checkpoints/                   # model weights (gitignored)
├── datasets/                      # training/eval data (gitignored)
├── runs/                          # training output: configs, COCO dataset, LoRA checkpoints, val_stats.json
├── examples/                      # runnable example scripts
├── setup.sh                       # venv + CUDA-aware torch install
└── src/impact_team_2/
    ├── inference/                 # build_predictor, predict
    ├── train/
    │   ├── med_sam.py             # train_medsam3 entry point
    │   └── sam.py                 # HF Transformers SAM3 path (separate stack)
    ├── vendor/medsam3/            # vendored MedSAM3: lora_layers + train_sam3_lora_native
    └── visual/                    # evaluate, prediction grid, training curves, dice helpers
```

## Acknowledgments

- [MedSAM3](https://github.com/Joey-S-Liu/MedSAM3) — Liu et al., 2025
- [SAM3](https://github.com/facebookresearch/sam3) — Meta AI
- Spleen ultrasound dataset — [DivyanshuTak/Ultrasoud_Unet_Segmentation](https://github.com/DivyanshuTak/Ultrasoud_Unet_Segmentation)
