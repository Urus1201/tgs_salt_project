from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    train_dir: Path
    valid_dir: Path
    unlabeled_dir: Optional[Path] = None
    model_dir: Path = Path('models')
    log_dir: Path = Path('logs/runs')
    num_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_workers: int = 4
    device: str = 'cuda'
    fp16: bool = True
    consistency_weight: float = 20.0
    ema_decay: float = 0.999
    grad_clip: float = 1.0

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    in_channels: int = 4
    n_classes: int = 1
    initial_filters: int = 64
    dropout_rate: float = 0.2

@dataclass
class DataConfig:
    """Data processing configuration."""
    image_size: int = 128
    val_size: float = 0.1
    seed: int = 42
    depths_file: Optional[str] = None

config = {
    'training': TrainingConfig(
        train_dir=Path('/root/tgs_salt_project/data/train'),
        valid_dir=Path('/root/tgs_salt_project/data/valid'),
        unlabeled_dir=Path('/root/tgs_salt_project/data/test')
    ),
    'model': ModelConfig(),
    'data': DataConfig()
}
