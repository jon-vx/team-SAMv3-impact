"""End-to-end evaluation: baseline vs fine-tuned MedSAM3.

Builds two predictors — the public `lal-Joey/MedSAM3_v1` baseline and the
fine-tuned LoRA from `runs/medsam3_finetune/best_lora_weights.pt` (run
`examples/test_train.py` first to produce it). Runs both over the spleen
validation split, plots the per-image grids and the training loss curves,
prints the dice summaries side by side, and surfaces the worst failure cases
of the fine-tuned model.

If the fine-tuned checkpoint isn't present yet, the baseline-only path runs
and the script tells you what's missing.
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from _data import ensure_spleen_data
from impact_team_2.inference import build_predictor
from impact_team_2.visual import (
    evaluate,
    show_prediction_grid,
    show_training_curves,
    worst_dice,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "runs" / "medsam3_finetune"
FINETUNED_WEIGHTS = RUN_DIR / "best_lora_weights.pt"
TRAINING_STATS = RUN_DIR / "val_stats.json"

DATA_DIR = ensure_spleen_data()

images = np.load(DATA_DIR / "images.npz")["images"]
masks = np.load(DATA_DIR / "masks.npz")["masks"] > 0
images[:, 0:20, 25:275] = [16, 16, 16]
print(f"images: {images.shape} | masks: {masks.shape}")

train_idx, val_idx = train_test_split(range(images.shape[0]), test_size=0.2, random_state=42)

# --- baseline -------------------------------------------------------------
baseline = build_predictor()  # public lal-Joey/MedSAM3_v1
before = evaluate(baseline, images, masks, val_idx, prompt="spleen",
                  threshold=0.01, desc="baseline")
print("\n=== BEFORE fine-tuning ===")
for k, v in before["summary"].items():
    print(f"  {k}: {v}")
show_prediction_grid(
    images, masks, before["results"],
    title="Before fine-tuning",
    save_path=RUN_DIR / "before_finetuning.png",
)

# Free baseline GPU memory before building the fine-tuned predictor.
del baseline
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# --- fine-tuned -----------------------------------------------------------
if not FINETUNED_WEIGHTS.exists():
    print(
        f"\n[test_eval] Fine-tuned weights not found at {FINETUNED_WEIGHTS}. "
        f"Run examples/test_train.py first to produce them, then re-run this script."
    )
    raise SystemExit(0)

finetuned = build_predictor(FINETUNED_WEIGHTS)
after = evaluate(finetuned, images, masks, val_idx, prompt="spleen",
                 threshold=0.01, desc="finetuned")

print("\n=== AFTER fine-tuning ===")
for k, v in after["summary"].items():
    print(f"  {k}: {v}")

print("\n=== Delta (after - before) ===")
for k in ("mean_dice", "max_dice", "min_dice", "dice_gt_0.5", "dice_gt_0.3"):
    print(f"  {k}: {after['summary'][k] - before['summary'][k]:+.4f}")

show_prediction_grid(
    images, masks, after["results"],
    title="After fine-tuning",
    save_path=RUN_DIR / "after_finetuning.png",
)

# --- training curves ------------------------------------------------------
if TRAINING_STATS.exists():
    show_training_curves(
        TRAINING_STATS,
        title="MedSAM3 LoRA training",
        save_path=RUN_DIR / "training_curves.png",
    )

# --- failure cases of the fine-tuned model -------------------------------
worst = worst_dice(after, k=5)
print(f"\nWorst {len(worst)} cases (dice ascending):")
for idx in worst:
    d = after["dice"][list(after["results"].keys()).index(idx)]
    print(f"  image {idx}: dice={d:.3f}")
show_prediction_grid(
    images, masks, worst,
    title="Worst dice (fine-tuned)",
    save_path=RUN_DIR / "worst_dice.png",
)
