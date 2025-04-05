import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__, "logs/mean_teacher.log")

def update_ema_variables(
    model: nn.Module,
    ema_model: nn.Module,
    alpha: float
) -> None:
    """Update teacher model parameters using exponential moving average."""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    logger.debug(f"Updated EMA variables with alpha={alpha}")

class MeanTeacher:
    """Mean Teacher model for semi-supervised learning."""
    
    def __init__(self, model: nn.Module, alpha: float = 0.999) -> None:
        logger.info("Initializing Mean Teacher")
        self.model = model
        self.ema_model = copy.deepcopy(model)
        self.alpha = alpha
        
        # Ensure EMA model is in eval mode
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.detach_()
    
    def update(self, student_model: Optional[nn.Module] = None) -> None:
        """Update teacher model using EMA of student parameters."""
        if student_model is None:
            student_model = self.model
        update_ema_variables(student_model, self.ema_model, self.alpha)
        logger.debug("Updated teacher model")
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get teacher model predictions."""
        return self.ema_model(x)
        
    def train(self) -> None:
        """Set both models to train mode."""
        self.model.train()
        self.ema_model.train()
        
    def eval(self) -> None:
        """Set both models to evaluation mode."""
        self.model.eval()
        self.ema_model.eval()

def consistency_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
) -> torch.Tensor:
    """Calculate consistency loss between student and teacher predictions."""
    loss = F.mse_loss(student_out, teacher_out.detach())
    logger.debug(f"Consistency loss: {loss.item():.4f}")
    return loss
