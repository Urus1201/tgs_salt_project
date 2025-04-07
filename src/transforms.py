import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Lambda
from utils.logger import setup_logger
import numpy as np
from skimage.restoration import denoise_wavelet

logger = setup_logger(__name__, "logs/transforms.log")

def get_train_transforms() -> A.Compose:
    """Get training data augmentation pipeline.
    
    Returns:
        A.Compose: Composed training transforms
    """
    logger.debug("Creating training transforms")
    return A.Compose([
        Lambda(image=lambda img, **kwargs: wavelet_denoise(img)),
        A.Resize(128, 128),  # First resize to target size
        A.OneOf([
            A.RandomCrop(width=120, height=120, p=0.5),
            A.CenterCrop(width=120, height=120, p=0.5),
        ], p=0.3),
        A.Resize(128, 128),  # Resize back to target size
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def get_valid_transforms() -> A.Compose:
    """Get validation data transforms."""
    logger.debug("Creating validation transforms")
    return A.Compose([
        Lambda(image=lambda img, **kwargs: wavelet_denoise(img)),
        A.Resize(128, 128),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def get_unlabeled_transforms() -> A.Compose:
    """Get transforms for unlabeled data."""
    logger.debug("Creating unlabeled data transforms")
    return A.Compose([
        Lambda(image=lambda img, **kwargs: wavelet_denoise(img)),
        A.Resize(height=128, width=128),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def get_test_transforms() -> A.Compose:
    """Get basic transforms for inference."""
    logger.debug("Creating test transforms")
    return A.Compose([
        Lambda(image=lambda img, **kwargs: wavelet_denoise(img)),
        A.Resize(height=128, width=128),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def wavelet_denoise(image, wavelet='db4', mode='soft', rescale_sigma=True):
    """Apply wavelet denoising - often good for seismic data."""
    # Handle multi-channel images
    if len(image.shape) == 3 and image.shape[2] > 1:
        # Process each channel individually
        result = np.zeros_like(image, dtype=np.float64)
        for i in range(image.shape[2]):
            # Normalize channel
            channel_norm = image[:,:,i].astype(np.float64) / 255.0
            # Denoise channel
            denoised_channel = denoise_wavelet(channel_norm, wavelet=wavelet, 
                                              mode=mode, rescale_sigma=rescale_sigma)
            # Store denoised channel
            result[:,:,i] = denoised_channel * 255
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        # For single channel images
        image_norm = image.astype(np.float64) / 255.0
        denoised = denoise_wavelet(image_norm, wavelet=wavelet, 
                                  mode=mode, rescale_sigma=rescale_sigma)
        return np.clip(denoised * 255, 0, 255).astype(np.uint8)