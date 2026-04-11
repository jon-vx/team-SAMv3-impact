#!/usr/bin/env python3
import os, sys, pathlib
import numpy as np
from dotenv import load_dotenv
from PIL import Image as PILImage
from huggingface_hub import login

_repo = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_repo / ".env.local")
login(token=os.getenv("HF_TOKEN"))

from impact_team_2.inference._inference_sam3 import build_predictor

image_paths = (
    [pathlib.Path(p) for p in sys.argv[1:]]
    or sorted((_repo / "datasets/ultrasound_spleen").glob("scan_*.png"))
)

predictor = build_predictor()
out_dir = _repo / "runs"
out_dir.mkdir(exist_ok=True)

TEXT_PROMPT = "spleen ,organ, spleen organ in ultrasound, dark oval region ,hypoechoic mass"
PRIOR_BOX = [65, 88, 200, 232]

for img_path in image_paths:
    result = predictor(img_path, text_prompt=TEXT_PROMPT, box=PRIOR_BOX, threshold=0.5)
    print(f"{img_path.name} — detections: {result['num_detections']}, scores: {result['scores'].tolist() if result['scores'] is not None else None}")

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
