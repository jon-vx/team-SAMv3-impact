from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from typing import Optional

# this is an old way to make boxes which is kind of cheating
# will be deprecated soon...
def make_fake_box(mask: np.ndarray):
    # cast and make c-compatible
    binary_mask = (mask > 0).astype(np.uint8)
    binary_mask = np.ascontiguousarray(binary_mask)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        box = np.array([x, y, x + w, y + h]) # xyxy format
    else:
        box = np.array([0, 0, mask.shape[1], mask.shape[0]])

# sam3 wants bounding boxes as input so we need to couple our existing dataset with
# corresponding, precomputed bounding boxes
class SAM3Dataset(Dataset):
    def __init__(self, images_np: np.ndarray, masks_np: np.ndarray, boxes_np: Optional[np.ndarray] = None):
        self.images = images_np
        self.masks = masks_np
        self.boxes = boxes_np

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]

        if mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = np.squeeze(mask, axis=-1)
            elif mask.shape[-1] > 1:
                mask = mask[:, :, 0]
        elif mask.ndim != 2:
            raise ValueError(f"Unexpected mask dimension: {mask.ndim} (shape: {mask.shape})")

        # use opencv to make a bounding box from the mask
        if self.boxes is not None:
            box = self.boxes[index]
        else:
            box = make_fake_box(mask)

        return image, mask, box
        