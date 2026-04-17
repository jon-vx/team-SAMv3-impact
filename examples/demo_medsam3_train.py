"""
Fine-tune MedSAM3 LoRA on the spleen dataset under `datasets/`,
then visualize training curves and predictions on the validation split.
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from _data import ensure_spleen_data
from impact_team_2.train import train_medsam3, TrainingConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "runs" / "medsam3_finetune"
TRAINING_STATS = OUTPUT_DIR / "val_stats.json"

DATA_DIR = ensure_spleen_data()

images = np.load(DATA_DIR / "images.npz")["images"]
masks = np.load(DATA_DIR / "masks.npz")["masks"] > 0

images[:, 0:20, 25:275] = [16, 16, 16]

print(f"images: {images.shape} | masks: {masks.shape}")

best_weights = train_medsam3(
    images,
    masks,
    output_dir=OUTPUT_DIR,
    category="spleen",
    training_config=TrainingConfig(num_epochs=10, batch_size=4, learning_rate=1e-4),
)

print(f"Best LoRA weights: {best_weights}")

# --- visualize training curves --------------------------------------------
from impact_team_2.visual import (
    evaluate,
    show_prediction_grid,
    show_training_curves,
)

if TRAINING_STATS.exists():
    show_training_curves(
        TRAINING_STATS,
        title="MedSAM3 LoRA training",
        save_path=OUTPUT_DIR / "training_curves.png",
    )

# --- evaluate on val split and show prediction grid -----------------------
from impact_team_2.inference import build_predictor

_, val_idx = train_test_split(
    range(images.shape[0]), test_size=0.2, random_state=42
)

predictor = build_predictor(best_weights)
eval_out = evaluate(
    predictor, images, masks, val_idx,
    prompt="spleen", threshold=0.5, desc="fine-tuned eval",
)

print("\n=== Fine-tuned evaluation ===")
for k, v in eval_out["summary"].items():
    print(f"  {k}: {v}")

show_prediction_grid(
    images, masks, eval_out["results"],
    title="After fine-tuning",
    save_path=OUTPUT_DIR / "finetuned_predictions.png",
)
