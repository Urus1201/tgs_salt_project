import os
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from utils.logger import setup_logger

logger = setup_logger(__name__, "logs/prepare_data.log")

def create_train_val_split(
    data_dir: str,
    val_size: float = 0.1,
    seed: int = 42
) -> None:
    """Create train-validation split from data directory.
    
    Args:
        data_dir: Root data directory
        val_size: Validation set proportion
        seed: Random seed for reproducibility
    """
    logger.info(f"Creating train-val split with validation size {val_size}")
    
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'valid'
    
    try:
        # Create validation directory structure
        (val_dir / 'images').mkdir(parents=True, exist_ok=True)
        (val_dir / 'masks').mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = list((train_dir / 'images').glob('*.png'))
        image_ids = [f.stem for f in image_files]
        
        # Split into train and validation
        train_ids, val_ids = train_test_split(
            image_ids,
            test_size=val_size,
            random_state=seed
        )
        
        # Move validation files
        for img_id in val_ids:
            # Move image
            src_img = train_dir / 'images' / f'{img_id}.png'
            dst_img = val_dir / 'images' / f'{img_id}.png'
            shutil.move(str(src_img), str(dst_img))
            
            # Move mask
            src_mask = train_dir / 'masks' / f'{img_id}.png'
            dst_mask = val_dir / 'masks' / f'{img_id}.png'
            shutil.move(str(src_mask), str(dst_mask))
        
        logger.info(f"Created train-val split:")
        logger.info(f"Train samples: {len(train_ids)}")
        logger.info(f"Validation samples: {len(val_ids)}")
        
    except Exception as e:
        logger.error(f"Error creating train-val split: {str(e)}")
        raise

if __name__ == '__main__':
    create_train_val_split('../data')