# SAM3 Vanilla Inference

Untrained SAM3 (`facebook/sam3`) inference using HuggingFace Transformers. Takes a bounding box as input and returns segmentation masks. Same API pattern as MedSAM3.

## Setup

Follows the same setup as the main project. See [README.md](README.md) for full setup instructions.

Requires a HuggingFace token:

```bash
echo "HF_TOKEN=hf_xxxxxxxxxxxxx" > .env.local
```

Additional dependencies used by SAM3 (`pillow`, `opencv-python`) are already listed in `pyproject.toml`.

## Inference

```python
from impact_team_2.inference._inference_sam3 import build_predictor, predict

# Build a reusable predictor
predictor = build_predictor()
result = predictor(
    "image.png",
    text_prompt="spleen organ in ultrasound",
    box=[65, 88, 200, 232],   # [x1, y1, x2, y2]
    threshold=0.5,
)

# Or use the module-level predict (builds model lazily on first call)
result = predict("image.png", "spleen", box=[65, 88, 200, 232])
```

`image` can be a file path, a PIL Image, or a numpy array (HxW or HxWx3, uint8 or float).

### Output

```python
{
    "boxes":          np.ndarray,      # (N, 4) bounding boxes derived from masks
    "scores":         np.ndarray,      # (N,) detection confidence scores
    "masks":          np.ndarray,      # (N, H, W) bool masks
    "num_detections": int,
    "image":          PIL.Image,       # input image as RGB PIL Image
}
```

If no detections pass the threshold, all fields except `num_detections` and `image` are `None`.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `text_prompt` | required | Text description of the target region |
| `box` | `None` | Prior bounding box `[x1, y1, x2, y2]`. Strongly recommended for focused segmentation. |
| `threshold` | `0.5` | Detection confidence threshold |

## Example Script

```bash
venv/bin/python examples/test_sam3_inf.py
```

Runs SAM3 on the spleen ultrasound scans in `datasets/ultrasound_spleen/` and saves purple overlay PNGs to `runs/`.

You can also pass specific images:

```bash
venv/bin/python examples/test_sam3_inf.py path/to/image.png
```

## Differences from MedSAM3

| | SAM3 | MedSAM3 |
|---|---|---|
| Model | `facebook/sam3` (HuggingFace) | `lal-Joey/MedSAM3_v1` (native sam3 + LoRA) |
| Fine-tuned | No — vanilla weights | Yes — LoRA fine-tuned on medical images |
| Device | CUDA / MPS / CPU | CUDA / CPU |
| Import | `_inference_sam3` | `_inference_medsam3` (default via `inference`) |

## Post-processing

Each mask undergoes:
1. Morphological closing (7x7 kernel, 2 iterations) to fill small holes
2. Largest connected component selection to remove noise
