"""Baseline MedSAM3 inference over the spleen val split via the public API."""

from sklearn.model_selection import train_test_split

from _data import load_spleen_data

import impact_team_2 as I

images, masks = load_spleen_data()
_, val_idx = train_test_split(range(images.shape[0]), test_size=0.2, random_state=42)

out = I.evaluate(
    model_list=["MedSAM"],
    modes=["not_finetuned"],
    images=images[val_idx],
    ground_truth=masks[val_idx],
    prompt="spleen",
    threshold=0.5,
    save_overlays_dir="runs/medsam3_baseline_overlays",
    save_overlays_n="worst:5",
)

print("summary:", out["MedSAM/not_finetuned"]["summary"])
