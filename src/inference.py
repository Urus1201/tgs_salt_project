import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import TGSDataset
from transforms import get_valid_transforms
from model import UNetResNet

def verify_rle(rle_string):
    """
    Verify that an RLE string meets competition requirements:
    - Space-delimited pairs of numbers
    - Numbers are positive integers
    - Pairs are sorted
    - No overlapping ranges
    """
    if not rle_string:
        return False
        
    # Split into pairs
    numbers = list(map(int, rle_string.split()))
    if len(numbers) % 2 != 0:
        return False
        
    # Check pairs are positive
    if any(n <= 0 for n in numbers):
        return False
        
    # Check pairs are sorted and non-overlapping
    pairs = list(zip(numbers[::2], numbers[1::2]))
    for i in range(len(pairs)-1):
        curr_start, curr_len = pairs[i]
        next_start = pairs[i+1][0]
        if curr_start + curr_len > next_start:
            return False
            
    return True

def rle_decode(rle_string, shape=(101, 101)):
    """
    Decode RLE string to binary mask for verification.
    shape: tuple of (height, width)
    """
    total_pixels = shape[0] * shape[1]
    mask = np.zeros(total_pixels, dtype=np.uint8)
    
    if rle_string.strip() == '1 1':
        mask[0] = 1
        return mask.reshape(shape, order='F')
        
    numbers = list(map(int, rle_string.split()))
    pairs = list(zip(numbers[::2], numbers[1::2]))
    
    for start, length in pairs:
        # Convert 1-indexed to 0-indexed
        start -= 1
        if start + length > total_pixels:
            raise ValueError(f"Invalid RLE: position {start + length} exceeds image size {total_pixels}")
        mask[start:start+length] = 1
        
    return mask.reshape(shape, order='F')

def rle_encode(mask):
    """
    Convert a binary mask into run-length encoding following competition format.
    mask: 2D array, 1 = salt, 0 = no salt
    Returns: Space-delimited string of pairs.
    For example '1 3 10 5' means pixels 1,2,3,10,11,12,13,14 are salt
    """
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask, dtype=np.uint8)
    
    if mask.shape != (101, 101):
        raise ValueError(f"Mask must be 101x101, got {mask.shape}")
    
    # Handle empty mask case early
    if not np.any(mask):
        return '1 1'
        
    # Flatten mask in column-major order (top-to-bottom, then left-to-right)
    pixels = mask.flatten(order='F')
    
    # Find transitions
    diff = np.diff(np.concatenate([[0], pixels, [0]]))
    starts = np.where(diff == 1)[0] + 1  # +1 for 1-based indexing
    ends = np.where(diff == -1)[0]
    
    # Generate runs
    runs = np.stack([starts, ends - starts + 1], axis=1).flatten()
    
    # Convert to string
    rle = ' '.join(map(str, runs))
    
    # Verify encoding is correct
    try:
        decoded_mask = rle_decode(rle, shape=mask.shape)
        if not np.array_equal(decoded_mask, mask):
            print("Warning: RLE verification failed!")
            print(f"Original mask sum: {mask.sum()}")
            print(f"Decoded mask sum: {decoded_mask.sum()}")
            diff = mask != decoded_mask
            print(f"Number of differing pixels: {diff.sum()}")
            raise AssertionError("RLE encoding/decoding mismatch")
    except Exception as e:
        print(f"RLE verification error: {str(e)}")
        raise
        
    if not verify_rle(rle):
        raise ValueError("Generated RLE string does not meet competition format requirements")
    
    return rle

def mask_to_rle(img):
    """Convert mask to run-length encoding (RLE)"""
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def run_inference(
    test_dir,
    model_dir='models',
    output_file='submission.csv',
    batch_size=32,
    tta=True,
    device='cuda'
):
    # Load both student and teacher models
    student_model = UNetResNet().to(device)
    teacher_model = UNetResNet().to(device)
    
    student_model.load_state_dict(torch.load(Path(model_dir) / 'best_model.pth'))
    teacher_model.load_state_dict(torch.load(Path(model_dir) / 'best_teacher_model.pth'))
    
    student_model.eval()
    teacher_model.eval()

    # Initialize test dataset and dataloader
    test_dataset = TGSDataset(test_dir, transform=get_valid_transforms(), is_test=True)
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
                flipped_images = torch.flip(images, [-1])
                student_pred_flip = torch.flip(student_model(flipped_images), [-1])
                teacher_pred_flip = torch.flip(teacher_model(flipped_images), [-1])
                batch_preds.extend([student_pred_flip, teacher_pred_flip])

                # Vertical flip TTA
                flipped_images = torch.flip(images, [-2])
                student_pred_flip = torch.flip(student_model(flipped_images), [-2])
                teacher_pred_flip = torch.flip(teacher_model(flipped_images), [-2])
                batch_preds.extend([student_pred_flip, teacher_pred_flip])

            # Average all predictions
            batch_preds = torch.stack(batch_preds)
            batch_preds = torch.sigmoid(batch_preds)
            batch_preds = batch_preds.mean(0)

            # Convert to binary mask using optimal threshold
            batch_preds = (batch_preds > 0.5).float()

            # Convert to RLE format
            for pred, id_ in zip(batch_preds, ids):
                pred = pred.cpu().numpy().squeeze()
                rle = mask_to_rle(pred)
                predictions.append(rle)
                image_ids.append(id_)

    # Create submission DataFrame
    df = pd.DataFrame({
        'id': image_ids,
        'rle_mask': predictions
    })
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    run_inference(
        test_dir='../data/test',
        model_dir='../models',
        output_file='../submission.csv',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
