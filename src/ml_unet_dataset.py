import os
import glob
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import albumentations as A

class ReefPatchDataset(Dataset):
    """
    PyTorch Dataset for loading 1-channel Bathymetry images and 1-channel binary masks.
    """
    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        """
        Args:
            images_dir (str): Directory containing image .tif patches.
            masks_dir (str): Directory containing mask .tif patches.
            transform (albumentations.Compose): Optional data augmentation.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Collect all image files
        self.image_files = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
        self.mask_files = [os.path.join(masks_dir, os.path.basename(f)) for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def _normalize_depth(self, img_array):
        """
        Normalize the depth array (typically negative).
        We clamp the max depth (e.g. -50m to 0m) and scale to [0, 1].
        """
        # Remove NaNs if any exist
        img_array = np.nan_to_num(img_array, nan=0.0)
        
        # Min-Max Scaling (assuming depths are negative, e.g. -40m to 0m)
        min_depth, max_depth = -40.0, 0.0
        norm = (img_array - min_depth) / (max_depth - min_depth)
        return np.clip(norm, 0.0, 1.0)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        # Load Raster Image (Bathymetry)
        with rasterio.open(img_path) as src:
            image = src.read(1).astype(np.float32)
            
        # Load Mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)
            
        # Normalize image
        image = self._normalize_depth(image)
        
        # Albumentations requires HWC format for images
        # Since we have 1 channel, we expand dims
        image = np.expand_dims(image, axis=-1)

        # Apply transformations (e.g., Flips, Rotations)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Convert to PyTorch tensors (CHW format)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask

def get_training_augmentation():
    """
    Returns albumentations Compose object for training.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])

def get_validation_augmentation():
    """
    Returns albumentations Compose object for validation/testing (No augs).
    """
    return None
