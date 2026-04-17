"""Baseline vs fine-tuned MedSAM3 comparison over the spleen val split.

Runs `I.evaluate` on both modes, prints the summaries side by side with deltas,
plots the training curves (if `val_stats.json` exists), and surfaces the worst
fine-tuned failure cases.

Requires `runs/medsam3_finetune/best_lora_weights.pt` — run
`examples/demo_medsam3_train.py` first to produce it.
"""

from pathlib import Path

from sklearn.model_selection import train_test_split

from _data import load_spleen_data

import impact_team_2 as I
from impact_team_2.visual import show_prediction_grid, show_training_curves, worst_dice

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "runs" / "medsam3_finetune"
TRAINING_STATS = RUN_DIR / "val_stats.json"

images, masks = load_spleen_data()
_, val_idx = train_test_split(range(images.shape[0]), test_size=0.2, random_state=42)
val_images, val_masks = images[val_idx], masks[val_idx]

out = I.evaluate(
    model_list=["MedSAM"],
    modes=["not_finetuned", "finetuned"],
    images=val_images,
    ground_truth=val_masks,
    prompt="spleen",
    threshold=0.5,
    save_overlays_dir=RUN_DIR / "eval_overlays",
    save_overlays_n="all",
)

before = out["MedSAM/not_finetuned"]
after = out["MedSAM/finetuned"]

print("\n=== BEFORE fine-tuning ===")
for k, v in before["summary"].items():
    print(f"  {k}: {v}")

print("\n=== AFTER fine-tuning ===")
for k, v in after["summary"].items():
    print(f"  {k}: {v}")

print("\n=== Delta (after - before) ===")
for k in ("mean_dice", "max_dice", "min_dice", "dice_gt_0.5", "dice_gt_0.3"):
    print(f"  {k}: {after['summary'][k] - before['summary'][k]:+.4f}")

show_prediction_grid(
    val_images, val_masks, before["results"],
    title="Before fine-tuning", save_path=RUN_DIR / "before_finetuning.png",
)
show_prediction_grid(
    val_images, val_masks, after["results"],
    title="After fine-tuning", save_path=RUN_DIR / "after_finetuning.png",
)

if TRAINING_STATS.exists():
    show_training_curves(
        TRAINING_STATS,
        title="MedSAM3 LoRA training",
        save_path=RUN_DIR / "training_curves.png",
    )

worst = worst_dice(after, k=5)
print(f"\nWorst {len(worst)} cases (dice ascending):")
for idx in worst:
    print(f"  val[{idx}]: dice={after['dice'][idx]:.3f}")
show_prediction_grid(
    val_images, val_masks, worst,
    title="Worst dice (fine-tuned)",
    save_path=RUN_DIR / "worst_dice.png",
)
