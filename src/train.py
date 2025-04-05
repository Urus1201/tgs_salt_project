import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm

from model import UNetResNet
from dataset import TGSDataset
from transforms import get_train_transforms, get_valid_transforms
from losses import CombinedLoss
from mean_teacher import MeanTeacher, consistency_loss

def train_model(
    train_dir,
    valid_dir,
    model_dir='models',
    log_dir='logs/runs',
    num_epochs=100,
    batch_size=16,
    learning_rate=1e-4,
    num_workers=4,
    device='cuda',
    fp16=True,
    consistency_weight=20.0,
    ema_decay=0.999,
    grad_clip=1.0
):
    # Initialize directories
    model_dir = Path(model_dir)
    log_dir = Path(log_dir)
    model_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True, parents=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Initialize student model and mean teacher
    model = UNetResNet().to(device)
    teacher = MeanTeacher(model, alpha=ema_decay)

    # Initialize datasets and dataloaders
    train_dataset = TGSDataset(train_dir, transform=get_train_transforms())
    valid_dataset = TGSDataset(valid_dir, transform=get_valid_transforms())
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Initialize optimizer, losses and scaler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # First restart after 10 epochs
        T_mult=2  # Double the restart interval after each restart
    )
    criterion = CombinedLoss()
    scaler = GradScaler() if fp16 else None

    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        teacher.train()
        train_loss = 0
        train_supervised_loss = 0
        train_consistency_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass with mixed precision
            with autocast(enabled=fp16):
                student_pred = model(images)
                teacher_pred = teacher.predict(images)
                
                # Calculate supervised loss
                sup_loss = criterion(student_pred, masks)
                cons_loss = consistency_loss(student_pred, teacher_pred)
                loss = sup_loss + consistency_weight * cons_loss

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            if fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # Update teacher model
            teacher.update()
            
            # Update metrics
            train_loss += loss.item()
            train_supervised_loss += sup_loss.item()
            train_consistency_loss += cons_loss.item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'sup_loss': sup_loss.item(),
                'cons_loss': cons_loss.item()
            })

        # Calculate average training losses
        train_loss /= len(train_loader)
        train_supervised_loss /= len(train_loader)
        train_consistency_loss /= len(train_loader)

        # Validation phase
        model.eval()
        valid_loss = 0
        
        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to(device), masks.to(device)
                predictions = model(images)
                valid_loss += criterion(predictions, masks).item()
        
        valid_loss /= len(valid_loader)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/train_supervised', train_supervised_loss, epoch)
        writer.add_scalar('Loss/train_consistency', train_consistency_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # Save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), model_dir / 'best_model.pth')
            torch.save(teacher.ema_model.state_dict(), model_dir / 'best_teacher_model.pth')
            print(f'New best model saved! Validation Loss: {valid_loss:.4f}')

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} (Sup: {train_supervised_loss:.4f}, Cons: {train_consistency_loss:.4f})')
        print(f'Valid Loss: {valid_loss:.4f}, LR: {current_lr:.6f}')

    # Save final models
    torch.save(model.state_dict(), model_dir / 'final_model.pth')
    torch.save(teacher.ema_model.state_dict(), model_dir / 'final_teacher_model.pth')
    writer.close()

if __name__ == '__main__':
    train_model(
        train_dir='../data/train',
        valid_dir='../data/valid',
        model_dir='../models',
        log_dir='../logs/runs',
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
