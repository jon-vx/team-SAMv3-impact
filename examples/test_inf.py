"""Run baseline MedSAM3 inference over the spleen val split and show a grid."""

import numpy as np
from sklearn.model_selection import train_test_split

from _data import ensure_spleen_data
from impact_team_2.inference import predict
from impact_team_2.visual import evaluate, show_prediction_grid

DATA_DIR = ensure_spleen_data()

images = np.load(DATA_DIR / "images.npz")["images"]
masks = np.load(DATA_DIR / "masks.npz")["masks"] > 0

images[:, 0:20, 25:275] = [16, 16, 16]

print(f"images: {images.shape} | masks: {masks.shape}")
train_idx, val_idx = train_test_split(range(208), test_size=0.2, random_state=42)

eval_out = evaluate(predict, images, masks, val_idx, prompt="spleen", threshold=0.01)
print("summary:", eval_out["summary"])

show_prediction_grid(
    images, masks, eval_out["results"],
    title="MedSAM3 baseline (lal-Joey/MedSAM3_v1)",
    save_path="baseline_predictions.png",
)
