# data_preparation.py

import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.transform import from_bounds
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


class QuarryEdgeDataset(Dataset):
    """Датасет для DEM фрагментов карьера и масок бровок."""

    def __init__(
            self,
            dem_paths: List[Path],
            mask_paths: List[Path],
            transform: Optional[A.Compose] = None,
            normalize_type: str = "zscore"
    ):
        assert len(dem_paths) == len(mask_paths), "Количество DEM и масок должно совпадать!"

        self.dem_paths = sorted(dem_paths)
        self.mask_paths = sorted(mask_paths)
        self.transform = transform
        self.normalize_type = normalize_type

    def __len__(self):
        return len(self.dem_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Загружаем DEM
        with rasterio.open(self.dem_paths[idx]) as src:
            dem = src.read(1).astype(np.float32)

        # Загружаем маску
        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1).astype(np.float32)

        # Обрабатываем NaN
        dem = np.nan_to_num(dem, nan=0.0)
        mask = np.nan_to_num(mask, nan=0.0)

        # Нормализация DEM
        dem = self._normalize_dem(dem)

        # Бинаризация маски (0 или 1)
        mask = (mask > 0.5).astype(np.float32)

        # Аугментация
        if self.transform:
            transformed = self.transform(image=dem, mask=mask)
            dem = transformed['image']
            mask = transformed['mask']
        else:
            dem = torch.from_numpy(dem).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)

        # Убеждаемся, что mask имеет channel dimension
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return dem, mask

    def _normalize_dem(self, dem: np.ndarray) -> np.ndarray:
        """Нормализация DEM"""
        if self.normalize_type == "zscore":
            mean = np.mean(dem)
            std = np.std(dem)
            if std > 1e-6:
                dem = (dem - mean) / std

        elif self.normalize_type == "minmax":
            min_val = np.min(dem)
            max_val = np.max(dem)
            if max_val - min_val > 1e-6:
                dem = (dem - min_val) / (max_val - min_val)

        return dem


# ============================================================================
# АУГМЕНТАЦИЯ С МИНИМАЛЬНЫМ ШУМОМ
# ============================================================================

# def get_training_augmentation():
#     """Аугментация с минимальным шумом (0.1)."""
#     train_transform = A.Compose([
#         # Геометрические трансформации
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#
#         # Сдвиги, масштаб, поворот
#         A.Affine(
#             translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
#             scale=(0.9, 1.1),
#             rotate=(-45, 45),
#             mode=0,
#             p=0.5
#         ),
#
#         # Эластичные деформации
#         A.ElasticTransform(
#             alpha=50,
#             sigma=5,
#             p=0.2
#         ),
#
#         # МИНИМАЛЬНЫЙ шум для DEM
#         A.GaussNoise(
#             var_limit=(0.05, 0.1),  # Слабый шум
#             mean=0,
#             per_channel=False,
#             p=0.3
#         ),
#
#         ToTensorV2()
#     ])
#
#     return train_transform


def get_training_augmentation():
    """Аугментация для обучающей выборки с минимальным шумом."""
    train_transform = A.Compose([
        # Геометрические трансформации
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Сдвиги, масштаб, поворот (БЕЗ mode)
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=(-45, 45),
            p=0.5
        ),

        # Эластичные деформации
        A.ElasticTransform(
            alpha=50,
            sigma=5,
            p=0.2
        ),

        # МИНИМАЛЬНЫЙ шум (ИСПРАВЛЕНО)
        A.GaussNoise(
            var_limit=(0.05, 0.1),
            p=0.3
        ),

        ToTensorV2()
    ])

    return train_transform

def get_validation_augmentation():
    """Аугментация для валидационной выборки."""
    val_transform = A.Compose([
        ToTensorV2()
    ])

    return val_transform


# ============================================================================
# СОЗДАНИЕ DATALOADERS
# ============================================================================

def create_dataloaders(
        data_dir: Path,
        batch_size: int = 4,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        num_workers: int = 0,
        normalize_type: str = "zscore"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Создаёт DataLoader для train/val/test выборок."""

    # dem_files = sorted(data_dir.glob("bd_res_1m_*.tif"))
    # mask_files = sorted(data_dir.glob("line_res_1m_*.tif"))

    dem_files = sorted(data_dir.glob("aug_dem_*.tif"))
    mask_files = sorted(data_dir.glob("aug_mask_*.tif"))

    assert len(dem_files) == len(mask_files), "Количество DEM и масок не совпадает!"
    assert len(dem_files) > 0, f"Не найдено файлов в {data_dir}"

    print(f"✓ Найдено {len(dem_files)} пар (DEM + маска)")

    n_samples = len(dem_files)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_dem = dem_files[:n_train]
    train_mask = mask_files[:n_train]

    val_dem = dem_files[n_train:n_train + n_val]
    val_mask = mask_files[n_train:n_train + n_val]

    test_dem = dem_files[n_train + n_val:]
    test_mask = mask_files[n_train + n_val:]

    print(f"  Train: {len(train_dem)} | Val: {len(val_dem)} | Test: {len(test_dem)}")

    train_dataset = QuarryEdgeDataset(
        train_dem, train_mask,
        transform=get_training_augmentation(),
        normalize_type=normalize_type
    )

    val_dataset = QuarryEdgeDataset(
        val_dem, val_mask,
        transform=get_validation_augmentation(),
        normalize_type=normalize_type
    )

    test_dataset = QuarryEdgeDataset(
        test_dem, test_mask,
        transform=get_validation_augmentation(),
        normalize_type=normalize_type
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# СОХРАНЕНИЕ АУГМЕНТИРОВАННЫХ ДАННЫХ
# ============================================================================

def save_augmented_data(
        dataset: QuarryEdgeDataset,
        output_dir: Path,
        num_augmentations: int = 10,
        resolution: float = 1.0
):
    """Генерирует и сохраняет аугментированные данные."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nГенерация аугментированных данных:")
    print(f"  Исходных образцов: {len(dataset)}")
    print(f"  Аугментаций на образец: {num_augmentations}")
    print(f"  Всего будет создано: {len(dataset) * num_augmentations} пар")
    print(f"  Сохранение в: {output_dir.absolute()}\n")

    counter = 0

    for idx in tqdm(range(len(dataset)), desc="Обработка образцов"):
        for aug_idx in range(num_augmentations):
            dem, mask = dataset[idx]

            # Конвертируем в numpy
            dem_np = dem.squeeze().cpu().numpy().astype(np.float32)
            mask_np = mask.squeeze().cpu().numpy().astype(np.float32)

            # Имена файлов
            dem_filename = output_dir / f"aug_dem_{counter:04d}.tif"
            mask_filename = output_dir / f"aug_mask_{counter:04d}.tif"

            # Transform
            height, width = dem_np.shape
            transform = from_bounds(
                0, 0,
                width * resolution, height * resolution,
                width, height
            )

            # Сохраняем DEM
            with rasterio.open(
                    dem_filename, 'w',
                    driver='GTiff',
                    height=height, width=width, count=1,
                    dtype=dem_np.dtype,
                    transform=transform,
                    nodata=np.nan
            ) as dst:
                dst.write(dem_np, 1)

            # Сохраняем маску
            with rasterio.open(
                    mask_filename, 'w',
                    driver='GTiff',
                    height=height, width=width, count=1,
                    dtype=mask_np.dtype,
                    transform=transform,
                    nodata=0
            ) as dst:
                dst.write(mask_np, 1)

            counter += 1

    print(f"\n✓ Создано {counter} пар аугментированных данных")


# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================

def visualize_augmentation(
        dataset: QuarryEdgeDataset,
        num_samples: int = None,
        save_path: Optional[str] = None
):
    """Визуализирует примеры аугментации."""
    if num_samples is None or num_samples > len(dataset):
        num_samples = min(4, len(dataset))

    print(f"\nВизуализация {num_samples} образцов")

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(num_samples):
        dem, mask = dataset[i]

        dem_np = dem.squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()

        axes[i, 0].imshow(dem_np, cmap='terrain')
        axes[i, 0].set_title(f'DEM #{i + 1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title(f'Маска')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(dem_np, cmap='terrain')
        axes[i, 2].imshow(mask_np, cmap='Reds', alpha=0.5)
        axes[i, 2].set_title(f'Наложение')
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Сохранено в {save_path}")

    plt.show()


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    SOURCE_DIR = Path(__file__).parent.parent.parent / "data" / "1m" / "100"
    AUGMENTED_DIR = Path(__file__).parent.parent.parent / "data" / "1m" / "augmented"

    # Создаём DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=SOURCE_DIR,
        batch_size=2,
        # train_ratio=1,
        train_ratio=0.7,
        val_ratio=0.2,
        normalize_type="zscore"
    )

    # Проверяем батч
    print("\n" + "=" * 50)
    print("ПРОВЕРКА БАТЧА")
    print("=" * 50)
    dem_batch, mask_batch = next(iter(train_loader))
    print(f"DEM shape: {dem_batch.shape}")
    print(f"Mask shape: {mask_batch.shape}")
    print(f"DEM range: [{dem_batch.min():.2f}, {dem_batch.max():.2f}]")
    print(f"Mask unique: {torch.unique(mask_batch).tolist()}")

    # Визуализация
    print("\n" + "=" * 50)
    print("ВИЗУАЛИЗАЦИЯ")
    print("=" * 50)
    visualize_augmentation(train_loader.dataset, save_path="augmentation_examples.png")

    # Сохранение аугментированных данных
    print("\n" + "=" * 50)
    print("ГЕНЕРАЦИЯ ДАННЫХ")
    print("=" * 50)
    save_augmented_data(
        dataset=train_loader.dataset,
        output_dir=AUGMENTED_DIR,
        num_augmentations=10,
        resolution=1.0
    )

    print("\n✓ Готово!")
