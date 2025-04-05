import os
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_train_val_split(data_dir, val_size=0.1, seed=42):
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'valid'
    
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
    
    print(f'Created train-val split:')
    print(f'Train samples: {len(train_ids)}')
    print(f'Validation samples: {len(val_ids)}')

if __name__ == '__main__':
    create_train_val_split('../data')