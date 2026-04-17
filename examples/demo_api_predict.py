"""Single-image prediction showcase for `I.predict`.

Runs both models in baseline mode on one spleen image and writes a 4-panel
overlay (image | GT | pred | diff) per model to `runs/predict_demo/`.
"""

from pathlib import Path

from _data import load_spleen_data

import impact_team_2 as I
from impact_team_2.visual import dice_score, save_overlay

OUT_DIR = Path("runs/predict_demo")

images, masks = load_spleen_data()
image, gt = images[0], masks[0]

for model in ("SAM", "MedSAM"):
    pred = I.predict(image, prompt="spleen", model=model, mode="not_finetuned")
    d = dice_score(pred, gt) if pred is not None else 0.0
    save_overlay(
        image, gt, pred,
        OUT_DIR / f"{model.lower()}_baseline.png",
        dice=d, title=f"{model} baseline · image[0]",
    )
    print(f"{model}: dice={d:.3f} -> {OUT_DIR}/{model.lower()}_baseline.png")
