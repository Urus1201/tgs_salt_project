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

def rle_encode(mask):
    """
    Convert a binary mask into run-length encoding.
    Expects mask to be exactly (101, 101).
    """
    pixels = mask.flatten(order='F')
    # Pad with zeros at start and end
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def run_inference(
    test_dir,
    model_dir='models',
    output_file='submission.csv',
    batch_size=8,
    tta=True,
    device='cuda'
):
    # Load both student and teacher models
    student_model = UNetResNet(in_channels=4).to(device)
    teacher_model = UNetResNet(in_channels=4).to(device)

    student_model.load_state_dict(torch.load(Path(model_dir) / 'best_model.pth', map_location=device))
    teacher_model.load_state_dict(torch.load(Path(model_dir) / 'best_teacher_model.pth', map_location=device))
    
    student_model.eval()
    teacher_model.eval()

    # We assume your unlabeled test set is 4-channel in dataset too
    test_dataset = TGSDataset(
        root_dir=test_dir,
        transform=get_valid_transforms(),
        is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            batch_preds = []

            # Basic forward pass
            student_pred = student_model(images)
            teacher_pred = teacher_model(images)
            batch_preds.extend([student_pred, teacher_pred])

            if tta:
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
    df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")

if __name__ == '__main__':
    run_inference(
        test_dir='../data/test',
        model_dir='../models',
        output_file='../submission.csv',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
