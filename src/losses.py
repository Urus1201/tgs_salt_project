import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from typing import Tuple
from utils.logger import setup_logger

logger = setup_logger(__name__, "logs/losses.log")

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # BCE Loss
        bce_loss = self.bce(pred, target)
        
        # Dice Loss
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / \
                   (pred.sum() + target.sum() + self.smooth)
        
        # Combine losses
        return 0.5 * bce_loss + 0.5 * dice_loss

class TverskyLoss(nn.Module):
    """
    Tversky loss with beta parameter to handle class imbalance.
    Particularly useful for seismic data where salt deposits might be sparse.
    """
    def __init__(self, beta: float = 0.7, smooth: float = 1e-5) -> None:
        super().__init__()
        self.beta = beta
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (pred * target).sum()
        FP = (pred * (1 - target)).sum()
        FN = ((1 - pred) * target).sum()
        
        tversky = (TP + self.smooth) / (TP + self.beta * FP + \
                  (1 - self.beta) * FN + self.smooth)
        return 1 - tversky

class BoundaryLoss(nn.Module):
    """
    Special loss for seismic boundary detection.
    Puts more emphasis on salt deposit boundaries.
    """
    def __init__(self, theta: float = 3) -> None:
        super().__init__()
        self.theta = theta
        self.laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        if torch.cuda.is_available():
            self.laplacian_kernel = self.laplacian_kernel.cuda()
        
        # Extract boundaries using Laplacian
        pred = torch.sigmoid(pred)
        pred_boundary = F.conv2d(pred, self.laplacian_kernel, padding=1)
        target_boundary = F.conv2d(target, self.laplacian_kernel, padding=1)
        
        # Calculate weighted MSE loss
        boundary_weight = torch.exp(self.theta * torch.abs(target_boundary))
        loss = (boundary_weight * (pred_boundary - target_boundary) ** 2).mean()
        
        return loss

def get_combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    boundary_weight: float = 0.1
) -> torch.Tensor:
    """
    Combines multiple loss functions for optimal seismic segmentation.
    """
    bce_dice = BCEDiceLoss()
    tversky = TverskyLoss()
    boundary = BoundaryLoss()
    
    loss = bce_dice(pred, target) + \
           0.5 * tversky(pred, target) + \
           boundary_weight * boundary(pred, target)
    
    return loss

class CombinedLoss(nn.Module):
    """Combined loss function for salt segmentation."""
    
    def __init__(self, beta: float = 0.7, boundary_weight: float = 0.5) -> None:
        super().__init__()
        logger.info(f"Initializing CombinedLoss with beta={beta}")
        self.beta = beta
        self.boundary_weight = boundary_weight

    def binary_cross_entropy(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(pred, target)
    
    def dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        smooth = 1.0
        
        intersection = (pred * target).sum(dim=(2,3))
        union = (pred + target).sum(dim=(2,3))
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def tversky_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        smooth = 1.0
        
        TP = (pred * target).sum(dim=(2,3))
        FP = (pred * (1-target)).sum(dim=(2,3))
        FN = ((1-pred) * target).sum(dim=(2,3))
        
        tversky = (TP + smooth) / (TP + self.beta*FP + (1-self.beta)*FN + smooth)
        return 1 - tversky.mean()
    
    def boundary_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        # Use Laplacian kernel to detect edges
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        
        pred = torch.sigmoid(pred)
        pred_boundaries = F.conv2d(pred, laplacian_kernel, padding=1)
        target_boundaries = F.conv2d(target, laplacian_kernel, padding=1)
        
        boundary_bce = F.binary_cross_entropy_with_logits(
            pred_boundaries, 
            target_boundaries.detach()
        )
        return boundary_bce
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        bce_loss = self.binary_cross_entropy(pred, target)
        dice_loss = self.dice_loss(pred, target)
        tversky_loss = self.tversky_loss(pred, target)
        boundary_loss = self.boundary_loss(pred, target)
        
        total_loss = (
            bce_loss +
            dice_loss +
            tversky_loss +
            self.boundary_weight * boundary_loss
        )
        
        logger.debug(
            f"Losses - BCE: {bce_loss:.4f}, Dice: {dice_loss:.4f}, "
            f"Tversky: {tversky_loss:.4f}, Boundary: {boundary_loss:.4f}"
        )
        
        return total_loss

class ConsistencyLoss(nn.Module):
    """Mean Teacher consistency loss for semi-supervised learning"""
    def __init__(self, alpha: float = 0.999) -> None:
        super().__init__()
        self.alpha = alpha
    
    def forward(
        self,
        student_pred: torch.Tensor,
        teacher_pred: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(torch.sigmoid(student_pred), torch.sigmoid(teacher_pred))

def update_ema_variables(
    model: nn.Module,
    ema_model: nn.Module,
    alpha: float
) -> None:
    """Update teacher model parameters using exponential moving average"""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
