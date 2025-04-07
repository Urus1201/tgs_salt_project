import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import cv2
from tqdm import tqdm

from model import UNetResNet
from dataset import TGSDataset
from transforms import get_valid_transforms
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__, "logs/visualization.log")

def visualize_predictions(
    model_path: str = None, 
    num_samples: int = 10, 
    save_dir: str = None,
    teacher_model_path: str = None,
    cfg = None
):
    """
    Visualize model predictions vs ground truth for validation data.
    
    Args:
        model_path: Path to saved model weights
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations, if None will display
        teacher_model_path: Optional path to teacher model weights
        cfg: Configuration parameters
    """
    if cfg is None:
        cfg = config
    
    train_cfg = cfg['training']
    model_cfg = cfg['model']
    
    # Default paths if not provided
    if model_path is None:
        model_path = os.path.join(train_cfg.model_dir, 'best_model.pth')
    if teacher_model_path is None:
        teacher_model_path = os.path.join(train_cfg.model_dir, 'best_teacher_model.pth')
    
    device = torch.device(train_cfg.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize and load model
    student_model = UNetResNet(
        in_channels=model_cfg.in_channels,
        n_classes=model_cfg.n_classes
    ).to(device)
    student_model.load_state_dict(torch.load(model_path, map_location=device))
    student_model.eval()
    
    # Load teacher model if path provided
    teacher_model = None
    if os.path.exists(teacher_model_path):
        teacher_model = UNetResNet(
            in_channels=model_cfg.in_channels,
            n_classes=model_cfg.n_classes
        ).to(device)
        teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
        teacher_model.eval()
    
    # Load validation dataset
    valid_dataset = TGSDataset(
        root_dir=train_cfg.valid_dir,
        transform=get_valid_transforms()
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,  # Process one sample at a time for visualization
        shuffle=True,  # Randomly sample from validation set
        num_workers=0  # Set to 0 for debugging
    )
    
    # Create save directory if needed
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    # Get samples to visualize
    samples = []
    for i, (image, mask) in enumerate(valid_loader):
        if i >= num_samples:
            break
        samples.append((image, mask))
    
    # Visualize each sample
    for i, (image, mask) in enumerate(tqdm(samples, desc="Generating visualizations")):
        image = image.to(device)
        mask = mask.to(device)
        
        with torch.no_grad():
            # Get student model prediction
            student_pred = student_model(image)
            student_pred = torch.sigmoid(student_pred)
            student_binary = (student_pred > 0.5).float()
            
            # Get teacher model prediction if available
            teacher_pred = None
            teacher_binary = None
            if teacher_model:
                teacher_pred = teacher_model(image)
                teacher_pred = torch.sigmoid(teacher_pred)
                teacher_binary = (teacher_pred > 0.5).float()
        
        # Convert tensors to numpy for visualization
        image_np = image[0].cpu().numpy()
        
        # For visualization, we'll just use the RGB channels and not the depth
        # Convert from NCHW to HWC format for visualization
        image_rgb = np.transpose(image_np[:3], (1, 2, 0))
        
        # Normalize image for display (undo the normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_rgb = std * image_rgb + mean
        image_rgb = np.clip(image_rgb, 0, 1)
        
        # Get masks
        mask_np = mask[0, 0].cpu().numpy()
        student_pred_np = student_pred[0, 0].cpu().numpy()
        student_binary_np = student_binary[0, 0].cpu().numpy()
        
        # Create figure and plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Sample {i+1}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ground truth mask
        axes[0, 1].imshow(mask_np, cmap='gray')
        axes[0, 1].set_title('Ground Truth Mask')
        axes[0, 1].axis('off')
        
        # Student prediction
        axes[0, 2].imshow(student_pred_np, cmap='gray')
        axes[0, 2].set_title('Student Prediction (Raw)')
        axes[0, 2].axis('off')
        
        # Student binary prediction
        axes[1, 0].imshow(student_binary_np, cmap='gray')
        axes[1, 0].set_title('Student Prediction (Binary)')
        axes[1, 0].axis('off')
        
        # Overlay of ground truth and student prediction
        overlay = np.zeros((mask_np.shape[0], mask_np.shape[1], 3))
        # Ground truth in green
        overlay[..., 1] = mask_np
        # Prediction in red
        overlay[..., 0] = student_binary_np
        # Overlay (where both agree will appear yellow)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay (Green: GT, Red: Pred)')
        axes[1, 1].axis('off')
        
        # Teacher prediction if available
        if teacher_model:
            teacher_binary_np = teacher_binary[0, 0].cpu().numpy()
            axes[1, 2].imshow(teacher_binary_np, cmap='gray')
            axes[1, 2].set_title('Teacher Prediction (Binary)')
            axes[1, 2].axis('off')
        else:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save or display
        if save_dir:
            plt.savefig(save_path / f'sample_{i+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    logger.info(f"Visualization completed for {num_samples} samples")

def visualize_test_predictions(
    model_path: str = None,
    num_samples: int = 10,
    save_dir: str = None,
    cfg = None
):
    """
    Visualize model predictions on test data (no ground truth).
    
    Args:
        model_path: Path to saved model weights
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations, if None will display
        cfg: Configuration parameters
    """
    if cfg is None:
        cfg = config
    
    train_cfg = cfg['training']
    model_cfg = cfg['model']
    
    # Default path if not provided
    if model_path is None:
        model_path = os.path.join(train_cfg.model_dir, 'best_model.pth')
    
    device = torch.device(train_cfg.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize and load model
    model = UNetResNet(
        in_channels=model_cfg.in_channels,
        n_classes=model_cfg.n_classes
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load test dataset
    test_dataset = TGSDataset(
        root_dir=train_cfg.unlabeled_dir,
        transform=get_valid_transforms(),
        is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    # Create save directory if needed
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    # Get samples to visualize
    samples = []
    for i, (image, img_id) in enumerate(test_loader):
        if i >= num_samples:
            break
        samples.append((image, img_id))
    
    # Visualize each sample
    for i, (image, img_id) in enumerate(tqdm(samples, desc="Generating test visualizations")):
        image = image.to(device)
        
        with torch.no_grad():
            # Get model prediction
            pred = model(image)
            pred = torch.sigmoid(pred)
            binary = (pred > 0.5).float()
        
        # Convert tensors to numpy for visualization
        image_np = image[0].cpu().numpy()
        
        # For visualization, we'll just use the RGB channels and not the depth
        image_rgb = np.transpose(image_np[:3], (1, 2, 0))
        
        # Normalize image for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_rgb = std * image_rgb + mean
        image_rgb = np.clip(image_rgb, 0, 1)
        
        # Get prediction
        pred_np = pred[0, 0].cpu().numpy()
        binary_np = binary[0, 0].cpu().numpy()
        
        # Create figure and plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Test Sample {img_id[0]}', fontsize=16)
        
        # Original image
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Raw prediction
        axes[1].imshow(pred_np, cmap='gray')
        axes[1].set_title('Model Prediction (Raw)')
        axes[1].axis('off')
        
        # Binary prediction
        axes[2].imshow(binary_np, cmap='gray')
        axes[2].set_title('Model Prediction (Binary)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save or display
        if save_dir:
            plt.savefig(save_path / f'test_{img_id[0]}.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    logger.info(f"Test visualization completed for {num_samples} samples")

if __name__ == '__main__':
    # Visualize validation results (with ground truth)
    visualize_predictions(
        save_dir='../visualizations/validation'
    )
    
    # Visualize test results (no ground truth)
    visualize_test_predictions(
        save_dir='../visualizations/test'
    )