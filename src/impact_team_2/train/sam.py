# -*- coding: utf-8 -*-
"""sam3_spleen_finetune_mlx.py"""
from argparse import ArgumentParser

import os
import cv2
import numpy as np
from pathlib import Path

# MLX and PyTorch
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_vlm import load
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from impact_team_2.train import SAM3Dataset
from torchmetrics.segmentation import DiceScore

# load sam3 with mlx-vlm
model_path = "mlx-community/sam3-bf16" 
model, processor = load(model_path)

# metric logging
dice_metric = DiceScore(num_classes=2)

def _run(dataloader, optimizer, loss_and_grad_fn, train: bool = False):
    count = 0
    losses = []
    
    for batch_images, batch_masks, batch_boxes in dataloader:
        images_list = [img.numpy() for img in batch_images]
        boxes_list = batch_boxes.unsqueeze(1).numpy().tolist()
    
        inputs = processor(
            images=images_list,
            input_boxes=boxes_list,
            return_tensors="np"
        )
    
        mlx_inputs = {k: mx.array(v) for k, v in inputs.items()}
    
        target_masks_pt = F.interpolate(
            batch_masks.float().unsqueeze(1),
            size=(256, 256), 
            mode="nearest"
        )
        target_masks_mx = mx.array(target_masks_pt.numpy())
    
        (loss, predicted_logits), grads = loss_and_grad_fn(model, mlx_inputs, target_masks_mx)
    
        if train:
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
        
        preds_pt = torch.tensor(np.array(predicted_logits))
        targets_pt = torch.tensor(np.array(target_masks_mx))
        
        dice_metric.update(preds_pt, targets_pt)
        losses.append(loss.item())
        
        count += 1
        
        mode = 'train' if train else 'val'
        print(f"({mode}) [{count}/{len(dataloader)}] Loss: {loss.item():.4f}, Avg Loss: {sum(losses)/len(losses):.4f}")
    
    return sum(losses) / len(losses)

def train_sam(images_in: str, masks_in: str, model_out: str, epochs: int = 20, tv_split: float = 0.8):
    
    
    images = np.load(images_in)["images"]
    masks = np.load(masks_in)["masks"]
    dataset = SAM3Dataset(images, masks)
    
    train_size = round(len(dataset) * tv_split)
    val_size = len(dataset) - train_size
    
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False) 
    
    model.freeze()
    for name, module in model.named_modules():
        if "mask_decoder" in name:
            module.unfreeze()
    
    optimizer = optim.Adam(learning_rate=1e-4)
    
    def compute_loss(model_params, inputs_dict, target_masks_mx):
        outputs = model(**inputs_dict)
        predicted_logits = outputs.pred_masks[:, :1, :, :] 
        loss = mx.mean(nn.losses.binary_cross_entropy(predicted_logits, target_masks_mx))
        return loss, predicted_logits
    
    loss_and_grad_fn = nn.value_and_grad(model, compute_loss)
    
    for epoch in range(epochs):
        print(f"\n========== EPOCH {epoch + 1}/{epochs} ==========")
        
        # training phase
        model.train()
        _run(dataloader, optimizer, loss_and_grad_fn, train=True)
        train_dice = dice_metric.compute()
        print(f"(train) Epoch {epoch + 1} Dice Coefficient: {train_dice.item():.4f}")
        dice_metric.reset()
            
        # validation phase
        model.eval() # Good practice to switch modes
        # FIX 3: Pass val_dataloader and train=False
        _run(val_dataloader, optimizer, loss_and_grad_fn, train=False)
        val_dice = dice_metric.compute()
        print(f"(val) Epoch {epoch + 1} Dice Coefficient: {val_dice.item():.4f}")
        dice_metric.reset()
    
    # Save checkpoint after all epochs finish
    os.makedirs(model_out, exist_ok=True)
    model.save_weights(f"{model_out}/sam3_finetuned_weights.safetensors")
    processor.save_pretrained(model_out)
    print(f"\nTraining Complete. Model weights and processor successfully saved to: {model_out}")
    
    
if __name__ == "__main__":
    parser = ArgumentParser(prog="train/sam.py")
    parser.add_argument("images_in")
    parser.add_argument("masks_in")
    parser.add_argument("model_out")
    parser.add_argument("--tv-split", type=float)
    parser.add_argument("--epochs", type=int)
    
    args = parser.parse_args()
    train_sam(**vars(args))