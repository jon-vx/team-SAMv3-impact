#!/usr/bin/env python3
"""
SAM3 inference with automated bounding box generation via vendor UNet (INIA).

Usage:
  venv/bin/python examples/test_sam3_inf.py                          # default
  venv/bin/python examples/test_sam3_inf.py path/to/image.png        # specific image
  venv/bin/python examples/test_sam3_inf.py --unet path/to/model.h5  # custom weights
  venv/bin/python examples/test_sam3_inf.py --train                  # retrain UNet
"""
import os, argparse, pathlib
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PIL import Image as PILImage
from huggingface_hub import login

_repo = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_repo / ".env.local")
login(token=os.getenv("HF_TOKEN"))

from impact_team_2.vendor.team_one.INIA import fit, load_data, get_bboxes, plot_history, plot_predictions
from impact_team_2.inference._inference_sam3 import build_predictor, load_unet_weights

parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="*", help="Image paths (default: datasets/ultrasound_spleen/)")
parser.add_argument("--unet", default=str(_repo / "checkpoints/best_unetp.weights.h5"))
parser.add_argument("--train", action="store_true", help="Train a new UNet from scratch")
parser.add_argument("--threshold", type=float, default=0.5)
args = parser.parse_args()

image_paths = (
    [pathlib.Path(p) for p in args.images]
    or sorted((_repo / "datasets/ultrasound_spleen").glob("scan_*.png"))
)

_data_dir = _repo / "datasets/ultrasound_spleen"

# Train or load UNet
if args.train:
    X_train, y_train, X_test, y_test = load_data(
        images_path=str(_data_dir / "images.npz"),
        masks_path=str(_data_dir / "masks.npz"),
    )
    unet_model, history = fit("unet++", X_train, y_train)
    plot_history(history, "unet++")
    plot_predictions(unet_model, X_test, y_test, n=5)
    os.makedirs(_repo / "checkpoints", exist_ok=True)
    unet_model.save_weights(str(_repo / "checkpoints/best_unetp.weights.h5"))
    print("[UNet] Weights saved to checkpoints/best_unetp.weights.h5")
else:
    unet_model = load_unet_weights(args.unet)
    X_test = None
    if (_data_dir / "images.npz").exists():
        _, _, X_test, _ = load_data(
            images_path=str(_data_dir / "images.npz"),
            masks_path=str(_data_dir / "masks.npz"),
        )

# Visualize UNet bounding boxes on test samples
if X_test is not None:
    bboxes = get_bboxes(unet_model, X_test, threshold=args.threshold)
    fig, axes = plt.subplots(1, min(3, len(X_test)), figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(X_test[i].squeeze(), cmap="gray")
        if bboxes[i] is not None:
            x_min, y_min, x_max, y_max = bboxes[i]
            ax.add_patch(plt.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor="red", facecolor="none",
            ))
        ax.set_title(f"UNet bbox — sample {i}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# Run SAM3 with UNet-generated bounding boxes
predictor = build_predictor(unet_model=unet_model)
out_dir = _repo / "runs"
out_dir.mkdir(exist_ok=True)

TEXT_PROMPT = "spleen ,organ, spleen organ in ultrasound, dark oval region ,hypoechoic mass"

for img_path in image_paths:
    result = predictor(img_path, text_prompt=TEXT_PROMPT, threshold=args.threshold)
    print(f"{img_path.name} — detections: {result['num_detections']}, "
          f"scores: {result['scores'].tolist() if result['scores'] is not None else None}")

    if result["masks"] is None:
        continue

    best = result["masks"][int(result["scores"].argmax())]
    base = np.array(result["image"].convert("RGBA"))
    overlay = np.zeros_like(base)
    overlay[best] = [171, 71, 188, 140]
    alpha = overlay[:, :, 3:4] / 255.0
    base[:, :, :3] = (base[:, :, :3] * (1 - alpha) + overlay[:, :, :3] * alpha).astype(np.uint8)

    out_path = out_dir / f"sam3_{img_path.stem}_overlay.png"
    PILImage.fromarray(base, mode="RGBA").convert("RGB").save(out_path)
    print(f"  saved: {out_path}")
