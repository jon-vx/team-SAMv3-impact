from impact_team_2.inference import predict
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

images = np.load(DATASETS_DIR / "images.npz")["images"]
masks = np.load(DATASETS_DIR / "masks.npz")["masks"] > 0

images[:, 0:20, 25:275,] = [16,16,16]

print("images:", images.shape, "| masks:", masks.shape)
train_idx, val_idx = train_test_split(range(208), test_size=0.2, random_state=42)

prompt = "spleen"
all_results = {}

for i in val_idx:
    result = predict(images[i], prompt, threshold=0.01)
    all_results[i] = result
    top_score = result["scores"].max() if result["scores"] is not None else 0
    print(f"Image {i:3d}: {result['num_detections']} detections, top score: {top_score:.3f}")

