"""SAM3 detect-then-segment cascade: before/after the UNet++ bbox generator.

Three phases, all on the same held-out val split:
    1. Baseline SAM3 with text-only prompting (`sam_use_unet=False`).
    2. Train UNet++ on the train split and save
       `checkpoints/best_unetp.weights.h5`.
    3. Re-evaluate SAM3 with the UNet cascade active (`sam_use_unet=True`)
       so it picks up the freshly-written checkpoint.

Prints the two summaries side by side with deltas, so the contribution of
the UNet bbox stage is legible.

Requires the `unet` extra: `pip install -e ".[unet]"`.
"""

import os

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

from sklearn.model_selection import train_test_split

from _data import load_spleen_data

import impact_team_2 as I
from impact_team_2.vendor.team_one.INIA import fit, preprocess

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "checkpoints"
WEIGHTS_PATH = CKPT_DIR / "best_unetp.weights.h5"
OVERLAY_DIR = REPO_ROOT / "runs" / "sam3_unet_cascade"

images, masks = load_spleen_data()
train_idx, val_idx = train_test_split(
    range(images.shape[0]), test_size=0.2, random_state=42
)
train_images, train_masks = images[train_idx], masks[train_idx]
val_images, val_masks = images[val_idx], masks[val_idx]


# --- 1. baseline: text-only SAM3 ------------------------------------------
print("\n### 1. SAM3 baseline (text only)")
text_only = I.evaluate(
    model_list=["SAM"],
    modes=["not_finetuned"],
    images=val_images,
    ground_truth=val_masks,
    prompt="spleen",
    threshold=0.5,
    sam_use_unet=False,
    save_overlays_dir=OVERLAY_DIR / "text_only",
    save_overlays_n="worst:5",
)["SAM/not_finetuned"]

I.clear_cache()


# --- 2. train UNet++ bbox generator on the OUTER train split --------------
# `INIA.load_data` would shuffle + split the full dataset on its own (different
# seed/ratio than our 80/20), which would leak val frames into UNet training.
# Instead, replicate the load_data pipeline (crop top-24 equipment band → pad
# to 320x320 via preprocess) on the outer train slice only, so the 42-image
# outer val set stays unseen by everything.
print("\n### 2. Training UNet++ bbox generator (train split only)")
X_train, y_train = preprocess(train_images[:, 24:], train_masks[:, 24:])
print(f"unet train: {X_train.shape}")
unet_model, _ = fit("unet++", X_train, y_train)
CKPT_DIR.mkdir(exist_ok=True)
unet_model.save_weights(str(WEIGHTS_PATH))
print(f"UNet++ weights saved -> {WEIGHTS_PATH}")


# --- 3. SAM3 with UNet cascade --------------------------------------------
print("\n### 3. SAM3 with UNet cascade")
with_unet = I.evaluate(
    model_list=["SAM"],
    modes=["not_finetuned"],
    images=val_images,
    ground_truth=val_masks,
    prompt="spleen",
    threshold=0.5,
    sam_use_unet=True,
    save_overlays_dir=OVERLAY_DIR / "with_unet",
    save_overlays_n="worst:5",
)["SAM/not_finetuned"]


# --- 4. comparison --------------------------------------------------------
print("\n### 4. Comparison (SAM/text-only  vs  SAM/UNet-cascade)")
print(f"{'metric':<14}{'text_only':>14}{'with_unet':>14}{'delta':>14}")
print("-" * 56)
for m in ("n", "mean_dice", "max_dice", "min_dice", "dice_gt_0.5", "dice_gt_0.3"):
    b, a = text_only["summary"][m], with_unet["summary"][m]
    if isinstance(b, float):
        print(f"{m:<14}{b:>14.4f}{a:>14.4f}{a - b:>+14.4f}")
    else:
        print(f"{m:<14}{b:>14}{a:>14}{'-':>14}")
