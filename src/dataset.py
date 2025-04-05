import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class TGSDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_test = is_test
        
        self.img_dir = self.root_dir / 'images'
        if not is_test:
            self.mask_dir = self.root_dir / 'masks'
        
        self.ids = [f.stem for f in self.img_dir.glob('*.png')]
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        # Load image
        img_path = self.img_dir / f'{img_id}.png'
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.is_test:
            if self.transform:
                augmented = self.transform(image=img)
                img = augmented['image']
            return img, img_id
        
        # Load mask for training
        mask_path = self.mask_dir / f'{img_id}.png'
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
        # Ensure mask has shape [H, W] -> [1, H, W]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        elif len(mask.shape) == 3 and mask.shape[-1] == 1:
            mask = mask.permute(2, 0, 1)
        
        return img, mask
