import os
import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, Union
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__, "logs/dataset.log")

class TGSDataset(Dataset):
    """Dataset class for TGS Salt Identification Challenge.
    
    Args:
        root_dir: Root directory containing images and masks
        transform: Optional albumentations transforms
        is_test: Whether this is test set (no masks)
        depths_file: Optional path to depths.csv
        cfg: Configuration parameters
    """
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[object] = None,
        is_test: bool = False,
        depths_file: Optional[str] = None,
        cfg=None
    ) -> None:
        if cfg is None:
            cfg = config
            
        self.data_cfg = cfg['data']
        depths_file = depths_file or self.data_cfg.depths_file

        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_test = is_test

        self.img_dir = self.root_dir / 'images'
        if not is_test:
            self.mask_dir = self.root_dir / 'masks'

        self.ids = [f.stem for f in self.img_dir.glob('*.png')]

        # Load depth information
        if depths_file is None:
            depths_file = Path(root_dir).parent / 'depths.csv'
        self.depths = pd.read_csv(depths_file, index_col='id')

        logger.info(f"Initializing dataset from {root_dir}")
        logger.info(f"Found {len(self.ids)} images")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, str]]:
        img_id = self.ids[idx]

        # Load image
        img_path = self.img_dir / f'{img_id}.png'
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load depth and normalize
        depth = self.depths.loc[img_id, 'z']
        depth = np.full((img.shape[0], img.shape[1], 1), depth, dtype=np.float32)

        # Concatenate depth as an additional channel
        img = np.concatenate([img, depth], axis=-1)

        if self.is_test:
            logger.debug(f"Loading test image {img_id}")
            if self.transform:
                augmented = self.transform(image=img)
                img = augmented['image']
                # Ensure channels are in the correct order for PyTorch (N,C,H,W)
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                if len(img.shape) == 3 and img.shape[-1] == 4:
                    img = img.permute(2, 0, 1)
            return img, img_id

        # Load mask for training
        mask_path = self.mask_dir / f'{img_id}.png'
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        logger.debug(f"Loading train image and mask {img_id}")
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
            # Ensure channels are in the correct order for PyTorch (N,C,H,W)
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if len(img.shape) == 3 and img.shape[-1] == 4:
                img = img.permute(2, 0, 1)

        # Ensure mask has shape [H, W] -> [1, H, W]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        elif len(mask.shape) == 3 and mask.shape[-1] == 1:
            mask = mask.permute(2, 0, 1)

        return img, mask
