# # data_preparation.py
#
# import torch
# from torch.utils.data import Dataset, DataLoader
# import rasterio
# import numpy as np
# from pathlib import Path
# import matplotlib.pyplot as plt
# from typing import List, Tuple, Optional
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
#
#
# class QuarryEdgeDataset(Dataset):
#     """
#     Датасет для DEM фрагментов карьера и масок бровок.
#
#     Поддерживает:
#     - Загрузку GeoTIFF файлов
#     - Нормализацию DEM
#     - Геометрическую и фотометрическую аугментацию
#     - Обработку NaN значений
#     """
#
#     def __init__(
#             self,
#             dem_paths: List[Path],
#             mask_paths: List[Path],
#             transform: Optional[A.Compose] = None,
#             normalize_type: str = "zscore"  # "zscore", "minmax", "none"
#     ):
#         """
#         Args:
#             dem_paths: Список путей к DEM файлам
#             mask_paths: Список путей к маскам бровок
#             transform: Albumentations трансформации
#             normalize_type: Тип нормализации ("zscore", "minmax", "none")
#         """
#         assert len(dem_paths) == len(mask_paths), "Количество DEM и масок должно совпадать!"
#
#         self.dem_paths = sorted(dem_paths)
#         self.mask_paths = sorted(mask_paths)
#         self.transform = transform
#         self.normalize_type = normalize_type
#
#     def __len__(self):
#         return len(self.dem_paths)
#
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Загружаем DEM
#         with rasterio.open(self.dem_paths[idx]) as src:
#             dem = src.read(1).astype(np.float32)
#
#         # Загружаем маску
#         with rasterio.open(self.mask_paths[idx]) as src:
#             mask = src.read(1).astype(np.float32)
#
#         # Обрабатываем NaN
#         dem = np.nan_to_num(dem, nan=0.0)
#         mask = np.nan_to_num(mask, nan=0.0)
#
#         # Нормализация DEM
#         dem = self._normalize_dem(dem)
#
#         # Бинаризация маски (0 или 1)
#         mask = (mask > 0.5).astype(np.float32)
#
#         # Аугментация (применяется к обоим: dem и mask)
#         if self.transform:
#             transformed = self.transform(image=dem, mask=mask)
#             dem = transformed['image']
#             mask = transformed['mask']
#
#         # Если трансформ не включает ToTensorV2, конвертируем вручную
#         if not isinstance(dem, torch.Tensor):
#             dem = torch.from_numpy(dem).unsqueeze(0)  # [1, H, W]
#             mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
#
#         return dem, mask
#
#     def _normalize_dem(self, dem: np.ndarray) -> np.ndarray:
#         """Нормализация DEM"""
#         if self.normalize_type == "zscore":
#             # Z-score нормализация: (x - mean) / std
#             mean = np.mean(dem)
#             std = np.std(dem)
#             if std > 1e-6:
#                 dem = (dem - mean) / std
#
#         elif self.normalize_type == "minmax":
#             # Min-Max нормализация: (x - min) / (max - min)
#             min_val = np.min(dem)
#             max_val = np.max(dem)
#             if max_val - min_val > 1e-6:
#                 dem = (dem - min_val) / (max_val - min_val)
#
#         # "none" — без нормализации
#         return dem
#
#
# # ============================================================================
# # АУГМЕНТАЦИЯ
# # ============================================================================
#
# def get_training_augmentation():
#     """
#     Аугментация для обучающей выборки.
#     Включает геометрические и фотометрические преобразования.
#     """
#     train_transform = A.Compose([
#         # Геометрические трансформации (применяются к DEM и маске)
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.ShiftScaleRotate(
#             shift_limit=0.1,  # Сдвиг до 10%
#             scale_limit=0.1,  # Масштаб ±10%
#             rotate_limit=45,  # Поворот ±45°
#             border_mode=0,
#             p=0.5
#         ),
#
#         # Фотометрические трансформации (только для DEM)
#         A.OneOf([
#             A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),  # Гауссовский шум
#             A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),  # Мультипликативный шум
#         ], p=0.3),
#
#         A.RandomBrightnessContrast(
#             brightness_limit=0.2,
#             contrast_limit=0.2,
#             p=0.3
#         ),
#
#         # Эластичные деформации (имитируют небольшие искажения)
#         A.ElasticTransform(
#             alpha=50,
#             sigma=5,
#             alpha_affine=5,
#             border_mode=0,
#             p=0.2
#         ),
#
#         ToTensorV2()
#     ])
#
#     return train_transform
#
#
# def get_validation_augmentation():
#     """
#     Аугментация для валидационной выборки.
#     Только ToTensor, без преобразований.
#     """
#     val_transform = A.Compose([
#         ToTensorV2()
#     ])
#
#     return val_transform
#
#
# # ============================================================================
# # СОЗДАНИЕ DATALOADERS
# # ============================================================================
#
# def create_dataloaders(
#         data_dir: Path,
#         batch_size: int = 4,
#         train_ratio: float = 0.6,
#         val_ratio: float = 0.2,
#         num_workers: int = 0,
#         normalize_type: str = "zscore"
# ) -> Tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Создаёт DataLoader для train/val/test выборок.
#
#     Args:
#         data_dir: Директория с данными
#         batch_size: Размер батча
#         train_ratio: Доля обучающей выборки (0.6 = 60%)
#         val_ratio: Доля валидационной выборки (0.2 = 20%)
#         num_workers: Количество процессов для загрузки данных
#         normalize_type: Тип нормализации DEM
#
#     Returns:
#         train_loader, val_loader, test_loader
#     """
#     # Находим все файлы
#     dem_files = sorted(data_dir.glob("bd_res_1m_*.tif"))
#     mask_files = sorted(data_dir.glob("line_res_1m_*.tif"))
#
#     assert len(dem_files) == len(mask_files), "Количество DEM и масок не совпадает!"
#     assert len(dem_files) > 0, f"Не найдено файлов в {data_dir}"
#
#     print(f"Найдено {len(dem_files)} пар (DEM + маска)")
#
#     # Разделяем на train/val/test
#     n_samples = len(dem_files)
#     n_train = int(n_samples * train_ratio)
#     n_val = int(n_samples * val_ratio)
#
#     train_dem = dem_files[:n_train]
#     train_mask = mask_files[:n_train]
#
#     val_dem = dem_files[n_train:n_train + n_val]
#     val_mask = mask_files[n_train:n_train + n_val]
#
#     test_dem = dem_files[n_train + n_val:]
#     test_mask = mask_files[n_train + n_val:]
#
#     print(f"Train: {len(train_dem)} | Val: {len(val_dem)} | Test: {len(test_dem)}")
#
#     # Создаём датасеты
#     train_dataset = QuarryEdgeDataset(
#         train_dem, train_mask,
#         transform=get_training_augmentation(),
#         normalize_type=normalize_type
#     )
#
#     val_dataset = QuarryEdgeDataset(
#         val_dem, val_mask,
#         transform=get_validation_augmentation(),
#         normalize_type=normalize_type
#     )
#
#     test_dataset = QuarryEdgeDataset(
#         test_dem, test_mask,
#         transform=get_validation_augmentation(),
#         normalize_type=normalize_type
#     )
#
#     # Создаём DataLoaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True
#     )
#
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )
#
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )
#
#     return train_loader, val_loader, test_loader
#
#
# # ============================================================================
# # ВИЗУАЛИЗАЦИЯ
# # ============================================================================
#
# def visualize_augmentation(
#         dataset: QuarryEdgeDataset,
#         num_samples: int = 4,
#         save_path: Optional[str] = None
# ):
#     """
#     Визуализирует примеры аугментации.
#
#     Args:
#         dataset: Датасет с аугментацией
#         num_samples: Количество примеров для отображения
#         save_path: Путь для сохранения изображения (опционально)
#     """
#     fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
#
#     if num_samples == 1:
#         axes = axes[np.newaxis, :]
#
#     for i in range(num_samples):
#         dem, mask = dataset[i]
#
#         # Конвертируем из тензора обратно в numpy
#         dem_np = dem.squeeze().numpy()
#         mask_np = mask.squeeze().numpy()
#
#         # DEM
#         axes[i, 0].imshow(dem_np, cmap='terrain')
#         axes[i, 0].set_title(f'DEM (образец {i + 1})')
#         axes[i, 0].axis('off')
#
#         # Маска
#         axes[i, 1].imshow(mask_np, cmap='gray')
#         axes[i, 1].set_title(f'Маска бровок')
#         axes[i, 1].axis('off')
#
#         # Наложение
#         axes[i, 2].imshow(dem_np, cmap='terrain')
#         axes[i, 2].imshow(mask_np, cmap='Reds', alpha=0.5)
#         axes[i, 2].set_title(f'DEM + маска')
#         axes[i, 2].axis('off')
#
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"Сохранено в {save_path}")
#
#     plt.show()
#
#
# # ============================================================================
# # ПРИМЕР ИСПОЛЬЗОВАНИЯ
# # ============================================================================
#
# if __name__ == "__main__":
#     # Путь к данным
#     DATA_DIR = Path("../../data/1m/100")
#
#     # Создаём DataLoaders
#     train_loader, val_loader, test_loader = create_dataloaders(
#         data_dir=DATA_DIR,
#         batch_size=2,
#         train_ratio=0.6,
#         val_ratio=0.2,
#         normalize_type="zscore"
#     )
#
#     # Проверяем один батч
#     print("\nПроверка батча:")
#     dem_batch, mask_batch = next(iter(train_loader))
#     print(f"DEM shape: {dem_batch.shape}")  # [B, 1, H, W]
#     print(f"Mask shape: {mask_batch.shape}")  # [B, 1, H, W]
#     print(f"DEM range: [{dem_batch.min():.2f}, {dem_batch.max():.2f}]")
#     print(f"Mask unique: {torch.unique(mask_batch)}")
#
#     # Визуализируем аугментацию
#     print("\nВизуализация аугментации:")
#     train_dataset = train_loader.dataset
#     visualize_augmentation(train_dataset, num_samples=4, save_path="augmentation_examples.png")

# data_preparation.py

import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class QuarryEdgeDataset(Dataset):
    """
    Датасет для DEM фрагментов карьера и масок бровок.
    """

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

        # Аугментация (применяется к обоим: dem и mask)
        if self.transform:
            transformed = self.transform(image=dem, mask=mask)
            dem = transformed['image']
            mask = transformed['mask']
        else:
            # Если нет трансформа, конвертируем вручную
            dem = torch.from_numpy(dem).unsqueeze(0)  # [1, H, W]
            mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        # ВАЖНО: убеждаемся, что mask имеет channel dimension
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]

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
# АУГМЕНТАЦИЯ (ИСПРАВЛЕНО)
# ============================================================================

def get_training_augmentation():
    """Аугментация для обучающей выборки."""
    train_transform = A.Compose([
        # Геометрические трансформации
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Используем Affine вместо ShiftScaleRotate
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=(-45, 45),
            mode=0,
            p=0.5
        ),

        # Фотометрические трансформации (только для DEM)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1.0),  # ИСПРАВЛЕНО
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.3),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),

        # Эластичные деформации (ИСПРАВЛЕНО)
        A.ElasticTransform(
            alpha=50,
            sigma=5,
            p=0.2
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

    # Находим все файлы
    dem_files = sorted(data_dir.glob("bd_res_1m_*.tif"))
    mask_files = sorted(data_dir.glob("line_res_1m_*.tif"))

    assert len(dem_files) == len(mask_files), "Количество DEM и масок не совпадает!"
    assert len(dem_files) > 0, f"Не найдено файлов в {data_dir}"

    print(f"Найдено {len(dem_files)} пар (DEM + маска)")

    # Разделяем на train/val/test
    n_samples = len(dem_files)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_dem = dem_files[:n_train]
    train_mask = mask_files[:n_train]

    val_dem = dem_files[n_train:n_train + n_val]
    val_mask = mask_files[n_train:n_train + n_val]

    test_dem = dem_files[n_train + n_val:]
    test_mask = mask_files[n_train + n_val:]

    print(f"Train: {len(train_dem)} | Val: {len(val_dem)} | Test: {len(test_dem)}")

    # Создаём датасеты
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

    # Создаём DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # ИСПРАВЛЕНО для MPS
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
# ВИЗУАЛИЗАЦИЯ (ИСПРАВЛЕНО)
# ============================================================================

def visualize_augmentation(
        dataset: QuarryEdgeDataset,
        num_samples: int = None,  # ИСПРАВЛЕНО: автоопределение
        save_path: Optional[str] = None
):
    """
    Визуализирует примеры аугментации.
    """
    # Автоматически ограничиваем num_samples размером датасета
    if num_samples is None or num_samples > len(dataset):
        num_samples = min(4, len(dataset))  # ИСПРАВЛЕНО

    print(f"Визуализация {num_samples} образцов из {len(dataset)} доступных")

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(num_samples):
        dem, mask = dataset[i]

        # Конвертируем из тензора обратно в numpy
        dem_np = dem.squeeze().cpu().numpy()  # ИСПРАВЛЕНО: добавлен .cpu()
        mask_np = mask.squeeze().cpu().numpy()

        # DEM
        axes[i, 0].imshow(dem_np, cmap='terrain')
        axes[i, 0].set_title(f'DEM (образец {i + 1})')
        axes[i, 0].axis('off')

        # Маска
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title(f'Маска бровок')
        axes[i, 1].axis('off')

        # Наложение
        axes[i, 2].imshow(dem_np, cmap='terrain')
        axes[i, 2].imshow(mask_np, cmap='Reds', alpha=0.5)
        axes[i, 2].set_title(f'DEM + маска')
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
    # Путь к данным
    DATA_DIR = Path(__file__).parent.parent.parent / "data" / "1m" / "100"

    # Создаём DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=2,
        train_ratio=0.6,
        val_ratio=0.2,
        normalize_type="zscore"
    )

    # Проверяем один батч
    print("\nПроверка батча:")
    dem_batch, mask_batch = next(iter(train_loader))
    print(f"DEM shape: {dem_batch.shape}")
    print(f"Mask shape: {mask_batch.shape}")
    print(f"DEM range: [{dem_batch.min():.2f}, {dem_batch.max():.2f}]")
    print(f"Mask unique: {torch.unique(mask_batch)}")

    # Визуализируем аугментацию (автоматически ограничится размером датасета)
    print("\nВизуализация аугментации:")
    train_dataset = train_loader.dataset
    visualize_augmentation(train_dataset, save_path="augmentation_examples.png")
