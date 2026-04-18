"""Shared spleen dataset bootstrap for the example scripts."""

import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "datasets" / "ultrasound_spleen"

DATA_URLS = {
    "images.npz": "https://github.com/DivyanshuTak/Ultrasoud_Unet_Segmentation/raw/refs/heads/main/images.npz",
    "masks.npz": "https://github.com/DivyanshuTak/Ultrasoud_Unet_Segmentation/raw/refs/heads/main/masks.npz",
}

# Burned-in acquisition banner at the top of every spleen ultrasound frame.
# Painted over with a near-black fill so SAM/MedSAM don't latch onto the text.
_BANNER_SLICE = (slice(None), slice(0, 20), slice(25, 275))
_BANNER_FILL = (16, 16, 16)


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


def load_spleen_data() -> tuple[np.ndarray, np.ndarray]:
    """Ensure the dataset is present, load it, and redact the banner.

    Returns:
        images: uint8 array of shape (N, H, W, 3) with the top-of-frame
            acquisition banner painted over.
        masks:  bool array of shape (N, H, W).
    """
    data_dir = ensure_spleen_data()
    images = np.load(data_dir / "images.npz")["images"]
    masks = np.load(data_dir / "masks.npz")["masks"] > 0

    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            f"expected spleen images with shape (N, H, W, 3), got {images.shape}"
        )
    if images.shape[1] < 20 or images.shape[2] < 275:
        raise ValueError(
            f"spleen images too small to redact banner: shape={images.shape}"
        )
    images[_BANNER_SLICE] = _BANNER_FILL
    return images, masks
