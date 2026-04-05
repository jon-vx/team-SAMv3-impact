from impact_team_2.train.data import make_fake_box
from torchmetrics import Metric, MetricCollection
from typing import cast, Literal
import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics import MeanMetric
from transformers import Sam3Model, Sam3Processor, pipeline
import torch
from torch.utils.tensorboard import SummaryWriter

# Initialize before your loop
writer = SummaryWriter(log_dir="runs/sam3_experiment")

from impact_team_2.train import SAM3Dataset

if torch.cuda.is_available():
    print("CUDA is available, using GPU.")
    device = torch.device("cuda")
elif torch.mps.is_available():
    print("MPS is available, using GPU.")
    device = torch.device("mps")
else:
    print("Warning: CUDA and MPS not available, using CPU instead.")
    device = torch.device("cpu")

tracked_metrics = MetricCollection([
    DiceScore(num_classes=2), 
    BinaryJaccardIndex(),
    MeanMetric(),
])

global_step = 0



def _run_epoch(model, processor, dataloader, optimizer, loss_fn, metrics: MetricCollection):
    global global_step
    
    for batch_images, batch_masks, batch_boxes in dataloader:
        optimizer.zero_grad() # clear gradients
        images_list = [img.numpy() for img in batch_images]
        boxes_list = batch_boxes.unsqueeze(1).tolist()
        target_masks = cast(torch.Tensor, batch_masks.float().unsqueeze(1).to(device))

        # I couldn't figure out how to train in batch so you get one at a time for now
        inputs = processor(
            images=images_list,
            input_boxes=boxes_list,
            return_tensors="pt"
        ).to(device)
        outputs = model(**inputs)

        predicted_logits = outputs.pred_masks
        predicted_logits = predicted_logits[:, :1, :, :] # this outputs probabilities I believe

        # rescale the output mask back to its original size (300, 300)
        predicted_logits_resized = F.interpolate(
            predicted_logits,
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # compute binary cross-entropy
        loss = loss_fn(predicted_logits_resized, target_masks)

        if model.training:
            loss.backward()
            optimizer.step()
            
        predicted_probs = torch.sigmoid(predicted_logits_resized)

        for k, v in metrics.items():
            if k == "avg_loss":
                step_loss = loss.item()
                metrics[k].update(step_loss)
                avg_loss = metrics[k].compute()
                writer.add_scalar("loss/step_loss", step_loss, global_step)
                writer.add_scalar("loss/avg_loss", avg_loss, global_step)
                print(f"{step_loss=}, {avg_loss=}")
                
            else:
                metrics[k].update(predicted_probs, target_masks.long()) # type:ignore
                step_val = metrics[k].compute()
                writer.add_scalar(f"other/{k}", step_val, global_step)
                
            
        
        global_step += 1

    return

tracked_metrics = MetricCollection({
    "dice": DiceScore(num_classes=2), 
    "iou": BinaryJaccardIndex(),
    "avg_loss": MeanMetric()
})

class SAMTrainer:
    def __init__(self, images: np.ndarray, masks: np.ndarray, boxes: np.ndarray, epochs: int = 20, lr = 1e-4):
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        
        # freeze weights other than mask decoder
        # saves vram
        for name, param in self.model.named_parameters():
            if "mask_decoder" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.dataset = SAM3Dataset(images, masks, boxes)
        self.dataloaders = {}
        self.dataloaders["train"] = DataLoader(self.dataset, batch_size=1, shuffle=True)
        
        self.epochs = epochs
        
        self.optimizer = Adam(params=[p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        self.metrics = {}
        self.metrics["train"] = tracked_metrics.clone().to(device)

    def make_split(self, splits: list[tuple[str, float, bool]]):
        total = len(self.dataset)
        amts = [round(total * s[1]) for s in splits]
        rem = total - sum(amts)

        ds_splits = torch.utils.data.random_split(self.dataset, [rem, *amts])
        self.dataloaders["train"] = DataLoader(ds_splits[0], batch_size=1, shuffle=True)
        for i in range(1, len(ds_splits)):
            ds = ds_splits[i]
            name, _, shuffle = splits[i-1]
            self.dataloaders[name] = DataLoader(ds, batch_size=1, shuffle=shuffle)
            self.metrics[name] = tracked_metrics.clone().to(device)

    def make_validation_split(self, amt: float):
        self.make_split([("val", amt, False)])

    def start(self):
        for split_k, split_dl in self.dataloaders.items():
            self.model.eval()
            if split_k == "train":
                self.model.train()
            
            for epoch in range(self.epochs):
                _run_epoch(self.model, self.processor, split_dl, self.optimizer, self.loss_fn, self.metrics[split_k])
                vals = self.metrics[split_k].compute()
                print(f"\n======= Epoch {epoch + 1}/{self.epochs} Summary =======")
                for k, v in vals.items():
                    print(f"{k}: {v}")
                
        print("Training Complete.")
                
    def save_model(self, out_path: str):
        os.makedirs(out_path, exist_ok=True)
        self.model.save_weights(f"{out_path}/sam3_finetuned_weights.safetensors")
        self.processor.save_pretrained(out_path)
        print(f"Model weights and processor successfully saved to: {out_path}")
    

def train_sam(images_in: str, masks_in: str, model_out: str, epochs: int = 20, val_split: float = 0.2, lr = 1e-4):
    images = np.load(images_in)["images"]
    masks = np.load(masks_in)["masks"]
    
    # you would replace this with the UNet bounding box generation #
    boxes = np.ones((masks.shape[0], 4))
    for i in range(masks.shape[0]):
        boxes[i] = make_fake_box(masks[i])
    
    trainer = SAMTrainer(images, masks, boxes, epochs, lr)
    trainer.make_validation_split(val_split)
    trainer.start()
    trainer.save_model(model_out)


if __name__ == "__main__":
    parser = ArgumentParser(prog="train/sam.py")
    parser.add_argument("images_in")
    parser.add_argument("masks_in")
    parser.add_argument("model_out")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=1)

    args = parser.parse_args()
    train_sam(**vars(args))
