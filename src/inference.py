import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F

from dataset import TGSDataset
from transforms import get_valid_transforms
from model import UNetResNet
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__, "logs/inference.log")

def rle_encode(mask: np.ndarray) -> str:
    """Convert binary mask to run-length encoding."""
    pixels = mask.flatten(order='F')
    # Pad with zeros at start and end
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def run_inference(cfg=None) -> None:
    """Run inference using config parameters."""
    if cfg is None:
        cfg = config
        
    train_cfg = cfg['training']
    model_cfg = cfg['model']
    
    logger.info("Starting inference with config...")
    
    # Load models
    student_model = UNetResNet(
        in_channels=model_cfg.in_channels,
        n_classes=model_cfg.n_classes
    ).to(train_cfg.device)
    teacher_model = UNetResNet(
        in_channels=model_cfg.in_channels,
        n_classes=model_cfg.n_classes
    ).to(train_cfg.device)

    student_model.load_state_dict(
        torch.load(Path(train_cfg.model_dir) / 'best_model.pth', map_location=train_cfg.device)
    )
    teacher_model.load_state_dict(
        torch.load(Path(train_cfg.model_dir) / 'best_teacher_model.pth', map_location=train_cfg.device)
    )
    
    logger.info("Models loaded successfully")

    student_model.eval()
    teacher_model.eval()

    # We assume your unlabeled test set is 4-channel in dataset too
    test_dataset = TGSDataset(
        root_dir=train_cfg.test_dir,
        transform=get_valid_transforms(),
        is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers
    )

    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(train_cfg.device)
            batch_preds = []

            # Basic forward pass
            student_pred = student_model(images)
            teacher_pred = teacher_model(images)
            batch_preds.extend([student_pred, teacher_pred])

            if train_cfg.tta:
                # Horizontal flip TTA
                flipped_images = torch.flip(images, dims=[-1])
                student_pred_flip = torch.flip(student_model(flipped_images), dims=[-1])
                teacher_pred_flip = torch.flip(teacher_model(flipped_images), dims=[-1])
                batch_preds.extend([student_pred_flip, teacher_pred_flip])

                # Vertical flip TTA
                flipped_images = torch.flip(images, dims=[-2])
                student_pred_flip = torch.flip(student_model(flipped_images), dims=[-2])
                teacher_pred_flip = torch.flip(teacher_model(flipped_images), dims=[-2])
                batch_preds.extend([student_pred_flip, teacher_pred_flip])

            # Average all predictions
            batch_preds = torch.stack(batch_preds, dim=0)
            batch_preds = torch.sigmoid(batch_preds)
            batch_preds = batch_preds.mean(dim=0)  # shape: (B, 1, 128, 128) if 1 channel

            # Convert to binary mask
            batch_preds = (batch_preds > 0.5).float()

            # Now we resize from (128,128) -> (101,101)
            # Because TGS competition expects exactly (101,101)
            batch_preds_101 = F.interpolate(
                batch_preds,
                size=(101, 101),
                mode='bilinear',
                align_corners=True
            )
            batch_preds_101 = (batch_preds_101 > 0.5).byte()

            # Convert to RLE format
            for pred, id_ in zip(batch_preds_101, ids):
                # pred shape: (1, 101, 101)
                pred = pred.squeeze().cpu().numpy()  # shape: (101, 101)
                rle = rle_encode(pred)
                predictions.append(rle)
                image_ids.append(id_)

    # Create submission DataFrame
    df = pd.DataFrame({
        'id': image_ids,
        'rle_mask': predictions
    })
    df.to_csv(train_cfg.output_file, index=False)
    logger.info(f"Predictions saved to {train_cfg.output_file}")

if __name__ == '__main__':
    run_inference(config)
