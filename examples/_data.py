"""Shared spleen dataset bootstrap for the example scripts."""

import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "datasets" / "ultrasound_spleen"

DATA_URLS = {
    "images.npz": "https://github.com/DivyanshuTak/Ultrasoud_Unet_Segmentation/raw/refs/heads/main/images.npz",
    "masks.npz": "https://github.com/DivyanshuTak/Ultrasoud_Unet_Segmentation/raw/refs/heads/main/masks.npz",
}


def ensure_spleen_data() -> Path:
    """Download images.npz / masks.npz into DATA_DIR if missing. Returns DATA_DIR."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in DATA_URLS.items():
        target = DATA_DIR / filename
        if target.exists():
            continue
        print(f"Downloading {filename} -> {target}")
        urllib.request.urlretrieve(url, target)
    return DATA_DIR
