# `impact_team_2` — User Guide

End-to-end reference for fine-tuning and evaluating SAM3 / MedSAM3 on custom
segmentation data. The README has a quickstart; this document is the full
manual — what the library does, how it's structured, every public function
you're likely to call, and the footguns we've hit ourselves.

- [1. Install](#1-install)
- [2. Mental model](#2-mental-model)
- [3. Data expectations](#3-data-expectations)
- [4. The unified API (`impact_team_2 as I`)](#4-the-unified-api-impact_team_2-as-i)
- [5. Fine-tuning](#5-fine-tuning)
- [6. Visualization](#6-visualization)
- [7. Example scripts](#7-example-scripts)
- [8. The UNet cascade](#8-the-unet-cascade)
- [9. Troubleshooting / known pathologies](#9-troubleshooting--known-pathologies)
- [10. Module map](#10-module-map)

---

## 1. Install

Requires Python 3.12+ and, realistically, an NVIDIA GPU (training on CPU is
unusably slow).

```bash
git clone https://github.com/jon-vx/team-SAMv3-impact.git
cd team-SAMv3-impact
./setup.sh                 # or: source setup.sh  to activate the env in current shell
```

`setup.sh` creates (or updates) the `impact-team-2` conda env. On Linux +
NVIDIA with a CUDA 12.x driver it uses `environment.yml` (torch 2.7.x on the
cu126 wheel index + TF 2.19, the cuDNN 9 stack that keeps torch and TF from
colliding on plugin registration). Older drivers, Apple Silicon, and CPU-only
boxes branch to the matching torch/TF variants. The `[unet]` extra pulls in
TensorFlow, keras-unet-collection, and `nvidia-cuda-nvcc-cu12`. `setup.sh`
also installs `activate.d` / `deactivate.d` hooks in the env that add the
pip-installed NVIDIA wheel libs to `LD_LIBRARY_PATH` and point `XLA_FLAGS` at
`libdevice.10.bc` — without this, TF/UNet training crashes on GPU.

### HuggingFace token

Both SAM3 (`facebook/sam3`) and MedSAM3 (`lal-Joey/MedSAM3_v1`) are gated /
private. `impact_team_2` auto-loads `.env.local` on import:

```bash
echo "HF_TOKEN=hf_xxxxxxxxxxxxx" > .env.local
```

Without it, the first predictor build will fail on the model download.

### Verifying the install

```bash
conda activate impact-team-2
python -c "import impact_team_2 as I; print('OK')"
```

For a full smoke test, run `python examples/demo_api_predict.py` — it
downloads the spleen dataset, runs one inference per model, and writes two
PNGs under `runs/predict_demo/`.

---

## 2. Mental model

Two models. Both fine-tunable. One API.

|                   | SAM3                                            | MedSAM3                                       |
|-------------------|-------------------------------------------------|-----------------------------------------------|
| Backbone          | `facebook/sam3` (HuggingFace Transformers)      | `lal-Joey/MedSAM3_v1` (native sam3 + LoRA)    |
| Domain            | General-purpose, vanilla weights                | LoRA-adapted on medical imagery               |
| Prompt path       | Text only, or text + UNet-predicted bbox        | Text only (detection comes from the text head) |
| Fine-tune style   | Unfreeze mask decoder                           | LoRA adapters on vision / text / mask decoder |
| Finetuned ckpt    | `runs/sam3_finetune/sam3_finetuned_weights.safetensors` | `runs/medsam3_finetune/best_lora_weights.pt` |

Every operation lives at one of three layers:

1. **`impact_team_2.api`** — `I.predict` / `I.evaluate`. Model-agnostic;
   resolves `(model, mode)` to the right backend.
2. **`impact_team_2.inference`** — backend-specific predictor builders
   (`_inference_medsam3.py`, `_inference_sam3.py`). Call these when you need
   a predictor bound to a non-canonical checkpoint path.
3. **`impact_team_2.train`** — fine-tuning entry points (`train_medsam3`,
   `train_sam`). Both take in-memory `(images, masks)` arrays and write
   checkpoints to `output_dir`.

`impact_team_2.visual` supplies the evaluation / rendering helpers that all
three layers use.

---

## 3. Data expectations

Both trainers and both predictors share one input convention:

| Array    | Shape              | Dtype          | Notes |
|----------|--------------------|----------------|-------|
| `images` | `(N, H, W, 3)`     | `uint8`        | RGB. Predictors also accept file paths / PIL / 2-D grayscale. |
| `masks`  | `(N, H, W)`        | `bool` / 0-1   | Per-image binary mask. Trainers coerce via `> 0`. |

The repo's reference dataset is a 208-frame spleen ultrasound set:

```python
from examples._data import load_spleen_data
images, masks = load_spleen_data()
# images: (208, H, W, 3) uint8, with the acquisition banner redacted.
# masks:  (208, H, W) bool.
```

`load_spleen_data` is worth reading: it downloads images.npz/masks.npz from
the public Ultrasoud_Unet_Segmentation repo on first call, validates the
shape, and redacts the top-of-frame acquisition banner so the models don't
latch onto the burned-in text. If you bring your own data, match these two
shapes and it'll plug in.

### Train/val discipline

Every demo script in this repo uses one seeded split:

```python
from sklearn.model_selection import train_test_split
train_idx, val_idx = train_test_split(
    range(images.shape[0]), test_size=0.2, random_state=42,
)
```

Stick to that pattern when adding scripts — both `train_medsam3` and
`train_sam` use `random_state=42` defaults, so re-using the same outer split
keeps models from training on frames they're later evaluated against.

---

## 4. The unified API (`impact_team_2 as I`)

### `I.predict(image, prompt, *, model, mode, threshold, sam_use_unet, return_details)`

Single-image inference. Returns the best (highest-score) predicted mask as a
`bool` ndarray, or `None` if no detections cleared `threshold`.

| Arg              | Default             | Notes |
|------------------|---------------------|-------|
| `image`          | —                   | `str` path, PIL image, or ndarray (H,W) / (H,W,3). |
| `prompt`         | —                   | Text description (e.g. `"spleen"`). |
| `model`          | `"MedSAM"`          | `"SAM"` or `"MedSAM"`. |
| `mode`           | `"not_finetuned"`   | `"not_finetuned"` uses the public weights; `"finetuned"` loads the canonical path under `runs/`. |
| `threshold`      | `0.5`               | Detection-score cutoff. See §9 for why SAM text-only often needs `0.01`. |
| `sam_use_unet`   | `False`             | SAM only — route through the UNet cascade (requires `checkpoints/best_unetp.weights.h5`). |
| `return_details` | `False`             | `True` returns the full backend dict (boxes, all scores, all masks, PIL image). |

```python
import impact_team_2 as I

mask = I.predict("scan.png", prompt="spleen", model="MedSAM")          # public weights
mask = I.predict(img_arr,    prompt="spleen", model="SAM",
                 mode="finetuned", threshold=0.01)                      # finetuned SAM

details = I.predict(img, "spleen", return_details=True)
details["masks"]   # (N, H, W) bool — all detections above threshold
details["scores"]  # (N,) detection confidences
details["boxes"]   # (N, 4) xyxy
```

### `I.evaluate(*, model_list, images, ground_truth, modes, prompt, threshold, save_overlays_dir, save_overlays_n, sam_use_unet)`

Multi-model / multi-mode batch evaluation. Only one predictor is resident on
the GPU at a time — evicted before the next model is built — so you can run
all four `(SAM|MedSAM) × (not_finetuned|finetuned)` combinations on a single
constrained GPU without OOMing.

| Arg                   | Default                          | Notes |
|-----------------------|----------------------------------|-------|
| `model_list`          | —                                | Subset of `["SAM", "MedSAM"]`. |
| `images`              | —                                | `(N, H, W, 3)` uint8. Already val-sliced. |
| `ground_truth`        | —                                | `(N, H, W)` bool / 0-1. |
| `modes`               | `("not_finetuned",)`             | Any subset of `("not_finetuned", "finetuned")`. |
| `prompt`              | `"object"`                       | Text prompt. |
| `threshold`           | `0.5`                            | `float` applies to every model; `dict[Model, float]` overrides per-model (e.g. `{"SAM": 0.01, "MedSAM": 0.5}`). |
| `save_overlays_dir`   | `None`                           | If set, writes 4-panel overlays to `<dir>/<model>_<mode>/*.png`. |
| `save_overlays_n`     | `0`                              | Which overlays to save: `0` = none, `int N` = first N, `"all"`, `"worst:K"`, `"best:K"`. |
| `sam_use_unet`        | `False`                          | See §8. |

Returns a dict keyed by `"<model>/<mode>"`:

```python
out = I.evaluate(
    model_list=["SAM", "MedSAM"],
    modes=["not_finetuned", "finetuned"],
    images=val_images, ground_truth=val_masks,
    prompt="spleen",
    threshold={"SAM": 0.01, "MedSAM": 0.5},
    save_overlays_dir="runs/overlays",
    save_overlays_n="worst:5",
)
out["SAM/finetuned"]["summary"]   # {"n", "mean_dice", "max_dice", "min_dice", "dice_gt_0.5", "dice_gt_0.3", ...}
out["SAM/finetuned"]["dice"]      # list[float], per-image, in input order
out["SAM/finetuned"]["results"]   # dict[int, dict] — raw backend output per image (for overlays, contact sheets)
out["SAM/finetuned"]["all_scores"]# flat list of every detection score seen
```

### `I.clear_cache()`

Evicts the currently-resident predictor and empties the CUDA cache. Call
between training phases so the next model has the full GPU to itself — all
the demos do this.

---

## 5. Fine-tuning

### `train_medsam3(images, masks, output_dir, *, category, val_split, seed, lora_config, training_config, pretrained_lora)`

```python
from impact_team_2.train import train_medsam3, TrainingConfig, LoRAConfig

best_weights = train_medsam3(
    images, masks,
    output_dir="runs/medsam3_finetune",
    category="spleen",                                           # text prompt baked into COCO
    training_config=TrainingConfig(num_epochs=10, batch_size=4, learning_rate=1e-4),
    lora_config=LoRAConfig(rank=16, alpha=32),                   # optional
)
```

What actually happens under the hood:

1. `train_test_split(range(n), test_size=val_split, random_state=seed)` with
   defaults `(0.2, 42)` — matches the outer demo split when you pass the full
   dataset.
2. The `(train_idx, val_idx)` split is written to
   `runs/medsam3_finetune/dataset/{train,valid}/*.png` + a COCO annotation
   JSON, so the vendored MedSAM3 `SAM3TrainerNative` can load it.
3. LoRA warm-start: by default the public `lal-Joey/MedSAM3_v1` weights are
   loaded as the starting point. Pass `pretrained_lora=None` to start LoRA
   from scratch, or a path to a local `.pt` to warm-start from your own run.
4. Training runs with both vision + text + mask-decoder LoRA modules active.
5. Checkpoints: `best_lora_weights.pt` (lowest val loss) and
   `last_lora_weights.pt`. `val_stats.json` feeds `show_training_curves`.

**TrainingConfig defaults:** `batch_size=4, num_epochs=10, learning_rate=1e-4,
weight_decay=0.01, num_workers=0`.

**LoRAConfig defaults:** `rank=16, alpha=32, dropout=0.1`, all encoders +
decoder enabled.

### `train_sam(images, masks, *, output_dir, epochs, lr, val_split, seed, box_source, unet_model, text_prompt)`

```python
from impact_team_2.train.sam import train_sam

ckpt = train_sam(
    images, masks,
    output_dir="runs/sam3_finetune",
    epochs=10,
    lr=1e-4,
    val_split=0.1,
    box_source="none",                # "none" | "unet" | "gt"
    text_prompt="spleen",             # used when box_source="none"
)
```

**Only the mask decoder is unfrozen.** The output safetensors is a few MB and
loads on top of public SAM3 via `load_state_dict(..., strict=False)`. This
means the detection/score head *does not learn* your domain — important
implication in §9.

`box_source` controls what prompt SAM3 sees during training:

| `box_source` | What SAM3 sees                                                | Notes |
|--------------|----------------------------------------------------------------|-------|
| `"none"` (default) | Text prompt only (`text_prompt`)                         | SAM3 learns mask-from-text. |
| `"unet"`     | UNet-predicted bboxes, fall back to full-image box on miss      | Requires `unet_model=...` (a loaded Keras UNet). Matches the `sam_use_unet=True` inference path. |
| `"gt"`       | Tight box derived from the ground-truth mask                    | Upper-bound / sanity check. SAM3 never gets this signal at inference. |

Per-epoch metrics land in `runs/sam3_finetune/tb/` (TensorBoard). Best-val
dice checkpoint: `sam3_finetuned_weights.safetensors`. Last-epoch backup:
`last_sam3_finetuned_weights.safetensors`. If `val_split=0` / `val_split=1`,
the last checkpoint is promoted to the "best" slot.

### `train_sam_from_files(images_in, masks_in, model_out, ...)`

CLI-friendly wrapper — takes `.npz` paths instead of arrays. The module also
exposes a `__main__` so `python -m impact_team_2.train.sam ...` works.

---

## 6. Visualization

Everything under `impact_team_2.visual`:

### Per-case helpers

```python
from impact_team_2.visual import (
    best_mask, dice_score, resize_mask, summarize_dice,
    save_overlay, save_comparison_overlay, save_contact_sheet, resolve_save_indices,
)
```

| Function                   | What it gives you |
|----------------------------|-------------------|
| `best_mask(result)`        | Pick the highest-score mask from a backend result dict. Returns `None` if no detections. |
| `dice_score(pred, gt)`     | Binary dice on bool/0-1 arrays. |
| `resize_mask(mask, shape)` | Nearest-neighbor resize. Tolerates `(H, W)` or `(H, W, C)` target. |
| `summarize_dice(dice, scores=None)` | Reduce a dice list to the summary dict `I.evaluate` returns. |
| `save_overlay(img, gt, pred, out_path, *, dice, score, title)` | 4-panel PNG: image / GT (green) / pred (magenta) / diff (TP green, FN red, FP blue). |
| `save_comparison_overlay(img, gt, baseline_preds, finetuned_preds, out_path, *, baseline_dice, finetuned_dice, title)` | Grid: GT + per-model baseline row + per-model finetuned row. |
| `save_contact_sheet(images, gt_masks, pred_masks, out_path, *, dice_scores, cols, title)` | One tile per image — dataset-wide failure scan. |
| `resolve_save_indices(dice_scores, how)` | Map `how` (int / `"all"` / `"worst:K"` / `"best:K"`) to concrete indices. Use if you write your own save loop. |

### Dataset-level plots

```python
from impact_team_2.visual import (
    evaluate, show_prediction_grid, show_training_curves, worst_dice, best_dice,
)
```

- `evaluate(predictor, images, masks, indices, *, prompt, threshold, desc)` —
  the *lower-level* evaluator, parallel to `I.evaluate` but taking a bound
  predictor and an index list. Returns the same
  `{"results", "dice", "all_scores", "summary"}` shape. Use when you need a
  predictor from a custom checkpoint path.
- `show_prediction_grid(images, masks, results, *, title, save_path)` —
  3-column grid (image / GT / pred) per `results` entry. `results` is any
  dict keyed by image index → backend output (i.e. `out["results"]`, or the
  dict returned by `worst_dice` / `best_dice`).
- `show_training_curves(stats_path, *, title, save_path)` — reads
  `val_stats.json` from `runs/medsam3_finetune/` and plots train/val dice +
  loss curves.
- `worst_dice(eval_out, k=5)` / `best_dice(eval_out, k=5)` — filter an
  `evaluate`-shaped dict down to the K worst/best-dice results, ready to
  hand back to `show_prediction_grid`.

Both `show_*` helpers save to `save_path` if the matplotlib backend is
non-interactive (headless) — so they "just work" on SSH'd dev boxes.

---

## 7. Example scripts

All under `examples/`, all driven by `_data.load_spleen_data()` so running
them in sequence downloads the dataset once.

| Script                               | What it does |
|--------------------------------------|---|
| `demo_api_predict.py`                | Smoke test: `I.predict` once per model on `images[0]`, writes overlays. |
| `demo_medsam3_train.py`              | `train_medsam3` on the full dataset (internal split matches outer), then plots training curves + predictions. |
| `demo_sam3_train.py`                 | `train_sam` with `--box-source {none,unet,gt}` and checkpoint sanity checks. Argparse CLI. |
| `demo_medsam3_eval.py`               | Baseline vs fine-tuned MedSAM3 with `I.evaluate`, training curves, worst-dice grid. Requires `best_lora_weights.pt` to already exist. |
| `demo_sam3_unet_cascade.py`          | Three phases on one val split: text-only SAM3 → train UNet++ on train slice only → SAM3 with UNet cascade → comparison table. |
| `demo_api.py`                        | End-to-end showcase: baseline eval → finetune both models → evaluate → comparison table + per-image grids + contact sheets. Reference for how the API is meant to be used. |

```bash
conda activate impact-team-2

# Quickest smoke test
python examples/demo_api_predict.py

# Full MedSAM3 flow
python examples/demo_medsam3_train.py
python examples/demo_medsam3_eval.py

# Full SAM3 flow with UNet cascade
python examples/demo_sam3_unet_cascade.py          # also trains the UNet
python examples/demo_sam3_train.py --box-source unet --unet checkpoints/best_unetp.weights.h5

# Unified end-to-end
python examples/demo_api.py
```

---

## 8. The UNet cascade

SAM3's text-only detection head doesn't know medical vocabulary, so on
prompts like `"spleen"` it returns near-zero scores. The detect-then-segment
cascade fixes this by letting a small UNet++ (the vendored INIA
implementation) predict a coarse segmentation, extracting the tightest
bounding box, and feeding *that box* to SAM3 as a geometric prompt. SAM3
then refines inside the box.

Three files matter:

- `checkpoints/best_unetp.weights.h5` — the canonical UNet weights path.
  `demo_sam3_unet_cascade.py` produces it; other scripts consume it when you
  pass `sam_use_unet=True`.
- `src/impact_team_2/inference/_inference_sam3.py::_box_from_unet` — the
  inference-time integration: grayscale-resize the image to 320×320, run the
  UNet, extract bbox, scale back to original image coordinates, hand to SAM3.
- `src/impact_team_2/vendor/team_one/INIA.py` — the vendored UNet++ trainer
  (`fit`, `load_data`, `preprocess`, `get_bboxes`, `plot_history`).

To use the cascade at inference:

```python
# Via the unified API
out = I.evaluate(
    model_list=["SAM"], modes=["not_finetuned"],
    images=val_images, ground_truth=val_masks,
    prompt="spleen", threshold=0.5,
    sam_use_unet=True,                  # ← picks up checkpoints/best_unetp.weights.h5
)

# Direct
from impact_team_2.inference._inference_sam3 import build_predictor
predictor = build_predictor(unet_weights="checkpoints/best_unetp.weights.h5")
result = predictor(img, text_prompt="spleen")
```

To use it at training time (`train_sam` with `box_source="unet"`), pass a
pre-loaded Keras UNet via `unet_model=...` — that way the trainer consumes
the same bboxes the inference path will see.

**Leakage warning:** the vendored `INIA.load_data` does its own shuffle +
28-image test split. If you train the UNet through that path and then
evaluate SAM3 on a *different* split (e.g. sklearn's `test_size=0.2`), the
UNet has seen some of your val frames during training.
`demo_sam3_unet_cascade.py` avoids this by preprocessing only the outer
train slice — copy that pattern when you plug in your own data.

---

## 9. Troubleshooting / known pathologies

### "Fine-tuned SAM3 reports dice=0.0 at inference"

Not a bug. `SAMTrainer` only unfreezes the mask decoder (see
`train/sam.py:70–71`), so the *detection score head* stays at base SAM3
weights. HF's `post_process_instance_segmentation(threshold=0.5)` filters
detections by that score before returning any mask — and on out-of-vocab
prompts like `"spleen"`, base SAM3's scores don't clear 0.5, so every
detection gets dropped (`masks == None` → dice appended as `0.0`).

Meanwhile the trainer's internal dice computed on raw logits is fine (will
happily report `val_dice=0.89` while inference reports 0.0) — that's the
signature of this issue.

Fix: pass a per-model threshold to `I.evaluate`:

```python
I.evaluate(..., threshold={"SAM": 0.01, "MedSAM": 0.5})
```

Or `threshold=0.01` as a scalar if you only care about SAM. `demo_api.py`
already does this.

### "Out of memory when running both models"

`I.evaluate` evicts the previous predictor before loading the next, so a
single `evaluate` call with `model_list=["SAM", "MedSAM"]` fits on constrained
GPUs (tested on a 35 GB H200 MIG slice). If you build predictors yourself
via `build_predictor(...)`, call `I.clear_cache()` between them — or keep
only one reference alive at a time and let Python's GC run before the next
allocation.

### "TF initializes before torch and grabs the whole GPU"

`impact_team_2/__init__.py` sets `TF_FORCE_GPU_ALLOW_GROWTH=true` and
`TF_GPU_ALLOCATOR=cuda_malloc_async` before anything else is imported, so
the fix is in place as long as you `import impact_team_2` (or anything from
it) before importing TF directly. If you import TF first, undo it with
those env vars set explicitly.

### "HF download fails with 401/403"

`HF_TOKEN` isn't set. Add it to `.env.local`. Token needs access to
`facebook/sam3` (SAM3 public but gated) and `lal-Joey/MedSAM3_v1`.

### "UNet training blows up with libdevice errors"

`setup.sh` installs a conda `activate.d` hook that sets
`XLA_FLAGS=--xla_gpu_cuda_data_dir=...` pointing at the `nvidia-cuda-nvcc-cu12`
wheel. If you built the env by hand, replicate that or rerun `setup.sh`.

### "Finetuned checkpoint not found"

Canonical paths are hard-coded in `api.py`:

- SAM: `runs/sam3_finetune/sam3_finetuned_weights.safetensors`
- MedSAM: `runs/medsam3_finetune/best_lora_weights.pt`

If yours are elsewhere, either move them into place, symlink, or use the
lower-level `build_predictor(weights_path=...)` directly rather than going
through `I.predict(mode="finetuned")`.

---

## 10. Module map

```
src/impact_team_2/
├── api.py                        # I.predict / I.evaluate / I.clear_cache
├── __init__.py                   # env setup, .env.local load, re-exports
├── __main__.py                   # `python -m impact_team_2`
├── inference/
│   ├── __init__.py               # predict / build_predictor re-export (MedSAM default)
│   ├── _inference_medsam3.py     # native sam3 + LoRA predictor
│   └── _inference_sam3.py        # HF Transformers SAM3 + UNet cascade glue
├── train/
│   ├── __init__.py               # train_medsam3, export_coco_dataset, LoRAConfig, TrainingConfig
│   ├── data.py                   # SAM3Dataset torch Dataset + make_fake_box
│   ├── med_sam.py                # train_medsam3 entry point (COCO export + vendored trainer)
│   └── sam.py                    # SAMTrainer (mask-decoder-only), train_sam, train_sam_from_files
├── vendor/
│   ├── medsam3/                  # vendored MedSAM3: lora_layers + train_sam3_lora_native
│   └── team_one/INIA.py          # vendored UNet++ bbox generator (load_data, fit, preprocess, get_bboxes)
├── visual/
│   ├── __init__.py               # flat re-exports
│   ├── utils.py                  # best_mask, dice_score, resize_mask, summarize_dice
│   ├── plot.py                   # evaluate, show_prediction_grid, show_training_curves, worst_dice, best_dice
│   └── overlays.py               # save_overlay, save_comparison_overlay, save_contact_sheet, resolve_save_indices
└── web/
    ├── __init__.py
    └── server.py                 # stub / WIP
```

Canonical on-disk artifacts:

```
checkpoints/
└── best_unetp.weights.h5         # UNet++ bbox generator (produced by demo_sam3_unet_cascade.py)
runs/
├── medsam3_finetune/
│   ├── best_lora_weights.pt
│   ├── last_lora_weights.pt
│   ├── val_stats.json            # consumed by show_training_curves
│   ├── tb/                       # TensorBoard logs
│   └── dataset/{train,valid}/    # COCO export written by train_medsam3
└── sam3_finetune/
    ├── sam3_finetuned_weights.safetensors
    ├── last_sam3_finetuned_weights.safetensors
    └── tb/
```

Anything under `checkpoints/`, `datasets/`, and `runs/` is gitignored.
