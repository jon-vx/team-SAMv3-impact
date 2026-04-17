"""Baseline SAM3 inference over the spleen val split via the public API.

Text-only prompting by default (`sam_use_unet=False`). Flip the flag to use
the UNet-cascade path (requires `checkpoints/best_unetp.weights.h5`).
"""

from sklearn.model_selection import train_test_split

from _data import load_spleen_data

import impact_team_2 as I

images, masks = load_spleen_data()
_, val_idx = train_test_split(range(images.shape[0]), test_size=0.2, random_state=42)

out = I.evaluate(
    model_list=["SAM"],
    modes=["not_finetuned"],
    images=images[val_idx],
    ground_truth=masks[val_idx],
    prompt="spleen",
    threshold=0.5,
    sam_use_unet=False,
    save_overlays_dir="runs/sam3_baseline_overlays",
    save_overlays_n="worst:5",
)

print("summary:", out["SAM/not_finetuned"]["summary"])
