import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def update_ema_variables(model, ema_model, alpha):
    """Update teacher model parameters using exponential moving average."""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

class MeanTeacher:
    def __init__(self, model, alpha=0.999):
        self.model = model
        self.ema_model = copy.deepcopy(model)
        self.alpha = alpha
        
        # Ensure EMA model is in eval mode
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.detach_()
    
    def update(self, student_model=None):
        """Update teacher model using EMA of student parameters."""
        if student_model is None:
            student_model = self.model
        update_ema_variables(student_model, self.ema_model, self.alpha)
    
    @torch.no_grad()
    def predict(self, x):
        """Get teacher model predictions."""
        return self.ema_model(x)
        
    def train(self):
        """Set both models to train mode."""
        self.model.train()
        self.ema_model.train()
        
    def eval(self):
        """Set both models to evaluation mode."""
        self.model.eval()
        self.ema_model.eval()

def consistency_loss(student_out, teacher_out, mask=None):
    """
    A simple L2 or MSE-based consistency loss between student and teacher predictions.
    """
    return F.mse_loss(student_out, teacher_out.detach())
