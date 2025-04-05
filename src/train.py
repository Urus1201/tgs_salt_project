import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import UNetResNet
from dataset import TGSDataset
from transforms import get_train_transforms, get_valid_transforms, get_unlabeled_transforms
from losses import CombinedLoss
from mean_teacher import MeanTeacher, consistency_loss
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__, "logs/training.log")

def train_model(cfg=None):
    """Train UNetResNet model with Mean Teacher approach.
    
    Args:
        cfg: Optional config dictionary. If None, uses default config.
    """
    if cfg is None:
        cfg = config
    
    train_cfg = cfg['training']
    model_cfg = cfg['model']
    
    logger.info("Initializing training with config...")
    
    # Initialize directories
    model_dir = Path(train_cfg.model_dir)
    log_dir = Path(train_cfg.log_dir)
    model_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True, parents=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Initialize student model and mean teacher
    model = UNetResNet(
        n_classes=model_cfg.n_classes,
        in_channels=model_cfg.in_channels
    ).to(train_cfg.device)  # 4 channels: RGB + Depth
    teacher = MeanTeacher(model, alpha=train_cfg.ema_decay)
    logger.info(f"Model initialized on device: {train_cfg.device}")

    # Initialize supervised (labeled) dataset/dataloader
    train_dataset = TGSDataset(train_cfg.train_dir, transform=get_train_transforms())
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True
    )

    # If we have a valid_dir, create that loader
    valid_loader = None
    if train_cfg.valid_dir:
        valid_dataset = TGSDataset(train_cfg.valid_dir, transform=get_valid_transforms())
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=True
        )

    # If unlabeled_dir is provided, create an unlabeled dataset
    unlabeled_loader = None
    if train_cfg.unlabeled_dir:
        # Use is_test=True so that the dataset doesn't expect a mask
        unlabeled_dataset = TGSDataset(
            train_cfg.unlabeled_dir,
            transform=get_unlabeled_transforms(),
            is_test=True
        )
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=train_cfg.num_workers,
            pin_memory=True
        )

    # Initialize optimizer, loss, scheduler, scaler
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,
        T_mult=2
    )
    criterion = CombinedLoss()
    scaler = GradScaler() if train_cfg.fp16 else None

    best_loss = float('inf')
    logger.info("Starting training loop...")

    for epoch in range(train_cfg.num_epochs):
        # Training phase
        model.train()
        teacher.train()
        train_loss = 0
        train_supervised_loss = 0
        train_consistency_loss_total = 0

        # We'll iterate over the labeled loader
        labeled_iter = iter(train_loader)
        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader else None

        steps_per_epoch = len(train_loader)
        if unlabeled_loader:
            # Optionally match or exceed number of unlabeled batches
            steps_per_epoch = max(steps_per_epoch, len(unlabeled_loader))

        pbar = tqdm(range(steps_per_epoch), desc=f'Epoch {epoch+1}/{train_cfg.num_epochs}')

        for _ in pbar:
            try:
                labeled_data, labeled_mask = next(labeled_iter)
            except StopIteration:
                # Re-start labeled loader if it runs out
                labeled_iter = iter(train_loader)
                labeled_data, labeled_mask = next(labeled_iter)

            labeled_data = labeled_data.to(train_cfg.device)
            labeled_mask = labeled_mask.to(train_cfg.device)

            # (Optional) fetch unlabeled batch
            if unlabeled_iter:
                try:
                    unlabeled_data, _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_data, _ = next(unlabeled_iter)
                unlabeled_data = unlabeled_data.to(train_cfg.device)
            else:
                unlabeled_data = None

            # Forward pass with mixed precision
            with autocast(enabled=train_cfg.fp16):
                # Supervised loss
                student_pred_labeled = model(labeled_data)
                sup_loss = criterion(student_pred_labeled, labeled_mask)

                # Consistency loss with unlabeled data
                cons_loss_val = 0.0
                if unlabeled_data is not None:
                    teacher_pred_ul = teacher.predict(unlabeled_data)
                    student_pred_ul = model(unlabeled_data)
                    cons_loss_val = consistency_loss(student_pred_ul, teacher_pred_ul)

                total_loss = sup_loss + train_cfg.consistency_weight * cons_loss_val

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            if train_cfg.fp16:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()

            # Update teacher model
            teacher.update()

            train_loss += total_loss.item()
            train_supervised_loss += sup_loss.item()
            train_consistency_loss_total += cons_loss_val if isinstance(cons_loss_val, float) else cons_loss_val.item()

            pbar.set_postfix({
                'total_loss': total_loss.item(),
                'sup_loss': sup_loss.item(),
                'cons_loss': cons_loss_val if isinstance(cons_loss_val, float) else cons_loss_val.item()
            })

        train_loss /= steps_per_epoch
        train_supervised_loss /= steps_per_epoch
        train_consistency_loss_total /= steps_per_epoch

        # Validation phase (if we have valid_loader)
        valid_loss = 0
        if valid_loader:
            model.eval()
            with torch.no_grad():
                for images, masks in valid_loader:
                    images, masks = images.to(train_cfg.device), masks.to(train_cfg.device)
                    preds = model(images)
                    valid_loss += criterion(preds, masks).item()
            valid_loss /= len(valid_loader)
        else:
            valid_loss = train_loss  # If no validation set is given

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Logging
        writer.add_scalar('Loss/train_total', train_loss, epoch)
        writer.add_scalar('Loss/train_supervised', train_supervised_loss, epoch)
        writer.add_scalar('Loss/train_consistency', train_consistency_loss_total, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        logger.info(f"Epoch {epoch+1}/{train_cfg.num_epochs}:")
        logger.info(
            f"  Train Loss: {train_loss:.4f} "
            f"(Sup: {train_supervised_loss:.4f}, Cons: {train_consistency_loss_total:.4f})"
        )
        logger.info(f"  Valid Loss: {valid_loss:.4f}, LR: {current_lr:.6f}")

        # Save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), model_dir / 'best_model.pth')
            torch.save(teacher.ema_model.state_dict(), model_dir / 'best_teacher_model.pth')
            logger.info(f'New best model saved! Validation Loss: {valid_loss:.4f}')

    logger.info("Training completed!")
    torch.save(model.state_dict(), model_dir / 'final_model.pth')
    torch.save(teacher.ema_model.state_dict(), model_dir / 'final_teacher_model.pth')
    writer.close()

if __name__ == '__main__':
    train_model(config)
