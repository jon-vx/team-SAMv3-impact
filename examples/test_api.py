"""End-to-end `impact_team_2` API showcase.

Flow:
  1. Evaluate vanilla SAM3 + base MedSAM3 on the held-out val split.
  2. Fine-tune both models on the train split.
  3. Evaluate the fine-tuned SAM3 + MedSAM3 on the same val split.
  4. Compare every (model, mode) summary side by side.

Held-out discipline is the caller's job: we split train/val with a fixed seed
and only ever train on `train_idx`, only ever evaluate on `val_idx`.
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from _data import load_spleen_data

import impact_team_2 as I  # noqa: E402
from impact_team_2.train import train_medsam3
from impact_team_2.train.sam import train_sam

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = REPO_ROOT / "runs"
SAM_DIR = RUNS / "sam3_finetune"
MEDSAM_DIR = RUNS / "medsam3_finetune"
PROMPT = "spleen"


# --- 0. data ----------------------------------------------------------------
images, masks = load_spleen_data()
print(f"images: {images.shape} | masks: {masks.shape}")

train_idx, val_idx = train_test_split(
    range(images.shape[0]), test_size=0.2, random_state=42
)
val_images, val_masks = images[val_idx], masks[val_idx]
train_images, train_masks = images[train_idx], masks[train_idx]

all_summaries: dict[str, dict] = {}


# --- 1. baseline: vanilla SAM3 + base MedSAM3 on val ------------------------
print("\n### 1. Evaluating vanilla SAM3 and base MedSAM3 on val split ###")
baseline = I.evaluate(
    model_list=["SAM", "MedSAM"],
    images=val_images,
    ground_truth=val_masks,
    modes=["not_finetuned"],
    prompt=PROMPT,
    threshold=0.01,
)
for key, res in baseline.items():
    all_summaries[key] = res["summary"]

I.clear_cache()


# --- 2. fine-tune MedSAM3 on train ------------------------------------------
print("\n### 2a. Fine-tuning MedSAM3 on train split ###")
medsam_ckpt = train_medsam3(
    images=train_images,
    masks=train_masks,
    output_dir=MEDSAM_DIR,
    category=PROMPT,
)
print(f"MedSAM3 LoRA weights -> {medsam_ckpt}")
I.clear_cache()


# --- 3. fine-tune SAM3 on train ---------------------------------------------
print("\n### 2b. Fine-tuning SAM3 on train split ###")
sam_ckpt = train_sam(
    train_images,
    train_masks,
    output_dir=SAM_DIR,
    val_split=0.1,
    epochs=1,
)
print(f"SAM3 finetuned weights -> {sam_ckpt}")
I.clear_cache()


# --- 4. evaluate fine-tuned SAM3 + MedSAM3 on val ---------------------------
print("\n### 3. Evaluating fine-tuned SAM3 and MedSAM3 on val split ###")
finetuned = I.evaluate(
    model_list=["SAM", "MedSAM"],
    images=val_images,
    ground_truth=val_masks,
    modes=["finetuned"],
    prompt=PROMPT,
    threshold=0.01,
)
for key, res in finetuned.items():
    all_summaries[key] = res["summary"]

I.clear_cache()


# --- 5. compare -------------------------------------------------------------
print("\n### 4. Comparison ###")
metric_keys = ("n", "mean_dice", "max_dice", "min_dice", "dice_gt_0.5", "dice_gt_0.3")
header = f"{'metric':<14}" + "".join(f"{k:>22}" for k in all_summaries)
print(header)
print("-" * len(header))
for m in metric_keys:
    row = f"{m:<14}"
    for key in all_summaries:
        v = all_summaries[key].get(m, "-")
        row += f"{v:>22.4f}" if isinstance(v, float) else f"{str(v):>22}"
    print(row)

# Deltas: finetuned - not_finetuned, per model
print("\nDelta (finetuned - not_finetuned):")
for model in ("SAM", "MedSAM"):
    b = all_summaries.get(f"{model}/not_finetuned")
    f = all_summaries.get(f"{model}/finetuned")
    if not (b and f):
        continue
    print(f"  {model}:")
    for m in ("mean_dice", "max_dice", "min_dice", "dice_gt_0.5", "dice_gt_0.3"):
        print(f"    {m}: {f[m] - b[m]:+.4f}")
