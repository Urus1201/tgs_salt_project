import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return A.Compose([
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

def get_valid_transforms():
    return A.Compose([
        A.Resize(128, 128),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def get_unlabeled_transforms():
    """
    Augmentations for unlabeled data.
    """
    return A.Compose([
        A.Resize(height=128, width=128),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.5,), std=(0.5,)),
    ])

def get_test_transforms():
    """
    Basic transforms for inference (no heavy augmentation).
    """
    return A.Compose([
        A.Resize(height=128, width=128),
        A.Normalize(mean=(0.5,), std=(0.5,)),
    ])
