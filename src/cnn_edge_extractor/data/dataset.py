from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


class QuarryEdgeDataset(Dataset):
    def __init__(self, dem_paths: List[Path], mask_paths: List[Path], transform: Optional[A.Compose] = None, normalize_type: str = "zscore"):
        assert len(dem_paths) == len(mask_paths), "DEM/mask counts must match"
        self.dem_paths = sorted(dem_paths)
        self.mask_paths = sorted(mask_paths)
        self.transform = transform
        self.normalize_type = normalize_type

    def __len__(self) -> int:
        return len(self.dem_paths)

    def _normalize_dem(self, dem: np.ndarray) -> np.ndarray:
        if self.normalize_type == "zscore":
            mean = np.mean(dem)
            std = np.std(dem)
            if std > 1e-6:
                dem = (dem - mean) / std
        elif self.normalize_type == "minmax":
            min_val, max_val = np.min(dem), np.max(dem)
            if max_val - min_val > 1e-6:
                dem = (dem - min_val) / (max_val - min_val)
        return dem

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with rasterio.open(self.dem_paths[idx]) as src:
            dem = src.read(1).astype(np.float32)
        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1).astype(np.float32)
        dem = np.nan_to_num(dem, nan=0.0)
        mask = (np.nan_to_num(mask, nan=0.0) > 0.5).astype(np.float32)
        dem = self._normalize_dem(dem)
        if self.transform:
            transformed = self.transform(image=dem, mask=mask)
            dem = transformed['image']
            mask = transformed['mask']
        else:
            dem = torch.from_numpy(dem).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return dem, mask


def get_training_augmentation() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, scale=(0.9, 1.1), rotate=(-45, 45), p=0.5),
        A.ElasticTransform(alpha=50, sigma=5, p=0.2),
        A.GaussNoise(var_limit=(0.05, 0.1), p=0.3),
        ToTensorV2(),
    ])


def get_validation_augmentation() -> A.Compose:
    return A.Compose([ToTensorV2()])


def create_dataloaders(data_dir: Path, batch_size: int = 4, train_ratio: float = 0.6, val_ratio: float = 0.2, num_workers: int = 0, normalize_type: str = "zscore"):
    train_dem = sorted((data_dir / 'train' / 'images').glob('*.tif'))
    train_mask = sorted((data_dir / 'train' / 'masks').glob('*.tif'))
    val_dem = sorted((data_dir / 'val' / 'images').glob('*.tif'))
    val_mask = sorted((data_dir / 'val' / 'masks').glob('*.tif'))
    test_dem = sorted((data_dir / 'test' / 'images').glob('*.tif'))
    test_mask = sorted((data_dir / 'test' / 'masks').glob('*.tif'))
    train_ds = QuarryEdgeDataset(train_dem, train_mask, get_training_augmentation(), normalize_type)
    val_ds = QuarryEdgeDataset(val_dem, val_mask, get_validation_augmentation(), normalize_type)
    test_ds = QuarryEdgeDataset(test_dem, test_mask, get_validation_augmentation(), normalize_type)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers),
    )
