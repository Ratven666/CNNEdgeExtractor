# predict_full_dem.py

import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from app.cnn_pytorch.train_unet import UNet


# Импортируем архитектуру модели



# ============================================================================
# ПРЕДСКАЗАНИЕ НА ПОЛНОМ DEM
# ============================================================================

class DEMPredictor:
    """
    Предсказывает бровки на полном DEM карьера.
    Работает по sliding window для больших файлов.
    """

    def __init__(self, model_path, device='mps', window_size=100, overlap=20):
        """
        Args:
            model_path: Путь к обученной модели (.pth)
            device: Устройство ('cuda', 'mps', 'cpu')
            window_size: Размер окна (должен совпадать с размером при обучении)
            overlap: Перекрытие окон в пикселях (для сглаживания стыков)
        """
        self.device = torch.device(device)
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap

        # Загружаем модель
        self.model = UNet(in_channels=1, out_channels=1).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ Модель загружена: {model_path}")
        print(f"  Эпоха: {checkpoint['epoch']}")
        print(f"  Val IoU: {checkpoint['val_iou']:.4f}")
        print(f"  Val F1: {checkpoint['val_f1']:.4f}")

    def normalize_dem(self, dem):
        """Z-score нормализация (как при обучении)"""
        mean = np.mean(dem)
        std = np.std(dem)
        if std > 1e-6:
            dem = (dem - mean) / std
        return dem

    def predict_tile(self, tile):
        """
        Предсказание для одного тайла.

        Args:
            tile: numpy array [H, W]
        Returns:
            prediction: numpy array [H, W] с вероятностями [0-1]
        """
        # Нормализация
        tile_norm = self.normalize_dem(tile)

        # Конвертируем в тензор [1, 1, H, W]
        tile_tensor = torch.from_numpy(tile_norm).float().unsqueeze(0).unsqueeze(0)
        tile_tensor = tile_tensor.to(self.device)

        # Предсказание
        with torch.no_grad():
            prediction = self.model(tile_tensor)

        # Возвращаем numpy [H, W]
        return prediction.cpu().squeeze().numpy()

    def predict_full_dem(self, dem_path, output_path, threshold=0.5):
        """
        Предсказывает бровки для полного DEM.

        Args:
            dem_path: Путь к входному DEM (GeoTIFF)
            output_path: Путь для сохранения результата
            threshold: Порог бинаризации (0.5 по умолчанию)
        """
        print(f"\nОбработка: {dem_path}")

        # Открываем DEM
        with rasterio.open(dem_path) as src:
            height = src.height
            width = src.width
            transform = src.transform
            crs = src.crs

            print(f"  Размер: {width} x {height}")
            print(f"  CRS: {crs}")

            # Создаём выходной массив
            prediction_full = np.zeros((height, width), dtype=np.float32)
            weight_map = np.zeros((height, width), dtype=np.float32)

            # Вычисляем количество окон
            n_rows = (height - self.window_size) // self.stride + 1
            n_cols = (width - self.window_size) // self.stride + 1
            total_windows = n_rows * n_cols

            print(f"  Окон: {n_rows} x {n_cols} = {total_windows}")
            print(f"  Размер окна: {self.window_size}x{self.window_size}")
            print(f"  Перекрытие: {self.overlap}px\n")

            # Обрабатываем по окнам
            pbar = tqdm(total=total_windows, desc="Предсказание")

            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    # Координаты окна
                    row_start = row_idx * self.stride
                    col_start = col_idx * self.stride

                    # Читаем окно
                    window = Window(
                        col_start, row_start,
                        self.window_size, self.window_size
                    )

                    tile = src.read(1, window=window).astype(np.float32)

                    # Обрабатываем NaN
                    tile = np.nan_to_num(tile, nan=0.0)

                    # Предсказание
                    pred_tile = self.predict_tile(tile)

                    # Взвешенное накопление (центр окна имеет больший вес)
                    weight = self._create_weight_mask(self.window_size)

                    prediction_full[
                        row_start:row_start + self.window_size,
                        col_start:col_start + self.window_size
                    ] += pred_tile * weight

                    weight_map[
                        row_start:row_start + self.window_size,
                        col_start:col_start + self.window_size
                    ] += weight

                    pbar.update(1)

            pbar.close()

            # Нормализуем по весам
            prediction_full = np.divide(
                prediction_full,
                weight_map,
                out=np.zeros_like(prediction_full),
                where=weight_map != 0
            )

            # Бинаризация
            prediction_binary = (prediction_full > threshold).astype(np.uint8)

            # ИСПРАВЛЕНИЕ: Создаём профиль БЕЗ nodata для uint8
            profile = src.profile.copy()
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw',
                nodata=None  # Убираем nodata для uint8
            )

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(prediction_binary, 1)

            # Также сохраняем вероятности (float32 поддерживает nodata=nan)
            prob_path = str(output_path).replace('.tif', '_prob.tif')
            profile_prob = src.profile.copy()
            profile_prob.update(
                dtype=rasterio.float32,
                count=1,
                compress='lzw'
            )

            with rasterio.open(prob_path, 'w', **profile_prob) as dst:
                dst.write(prediction_full.astype(np.float32), 1)

            print(f"\n✓ Бинарная маска сохранена: {output_path}")
            print(f"✓ Вероятности сохранены: {prob_path}")

            return prediction_binary, prediction_full

    def _create_weight_mask(self, size):
        """
        Создаёт весовую маску с гауссовым распределением.
        Центр окна имеет вес 1, края уменьшаются к 0.
        """
        center = size // 2
        y, x = np.ogrid[:size, :size]

        # Гауссова функция
        sigma = size / 4
        weight = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))

        return weight


# ============================================================================
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================================

def visualize_results(dem_path, prediction_path, save_path='result_visualization.png', zoom_box=None):
    """
    Визуализирует исходный DEM и предсказанные бровки.

    Args:
        dem_path: Путь к исходному DEM
        prediction_path: Путь к предсказанной маске
        save_path: Путь для сохранения визуализации
        zoom_box: Кортеж (row_start, row_end, col_start, col_end) для zoom
    """

    # Загружаем DEM
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        # Заменяем NaN на 0 для визуализации
        dem = np.nan_to_num(dem, nan=0.0)

    # Загружаем предсказание
    with rasterio.open(prediction_path) as src:
        prediction = src.read(1)

    # Применяем zoom если указан
    if zoom_box is not None:
        r1, r2, c1, c2 = zoom_box
        dem = dem[r1:r2, c1:c2]
        prediction = prediction[r1:r2, c1:c2]

    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Исходный DEM
    im1 = axes[0].imshow(dem, cmap='terrain')
    axes[0].set_title('Исходный DEM', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Предсказанные бровки
    axes[1].imshow(prediction, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Предсказанные бровки', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # 3. Наложение
    axes[2].imshow(dem, cmap='terrain')
    axes[2].imshow(prediction, cmap='Reds', alpha=0.6)
    axes[2].set_title('DEM + Бровки', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Визуализация сохранена: {save_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Параметры
    MODEL_PATH = Path("models/best_edge_extractor.pth")
    # DEM_PATH = Path("../../data/1m/grib_1m.tif")  # Путь к полному DEM
    DEM_PATH = Path("../../data/grib_05m.tif")  # Путь к полному DEM
    OUTPUT_PATH = Path("output/predicted_edges.tif")

    # Создаём директорию для результатов
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ПРЕДСКАЗАНИЕ БРОВОК НА ПОЛНОМ DEM")
    print("=" * 70)

    # Проверяем наличие файлов
    if not MODEL_PATH.exists():
        print(f"❌ Модель не найдена: {MODEL_PATH}")
        return

    if not DEM_PATH.exists():
        print(f"❌ DEM не найден: {DEM_PATH}")
        return

    # Создаём предиктор
    predictor = DEMPredictor(
        model_path=MODEL_PATH,
        device='mps',  # Или 'cuda', 'cpu'
        window_size=100,
        overlap=20
    )

    # Предсказание
    prediction_binary, prediction_prob = predictor.predict_full_dem(
        dem_path=DEM_PATH,
        output_path=OUTPUT_PATH,
        threshold=0.5
    )

    # Статистика
    total_pixels = prediction_binary.size
    edge_pixels = np.sum(prediction_binary)
    edge_percentage = (edge_pixels / total_pixels) * 100

    print(f"\nСтатистика:")
    print(f"  Всего пикселей: {total_pixels:,}")
    print(f"  Пикселей бровок: {edge_pixels:,}")
    print(f"  Процент бровок: {edge_percentage:.2f}%")

    # Визуализация всего изображения
    print("\nГенерация визуализации (полный размер)...")
    visualize_results(
        dem_path=DEM_PATH,
        prediction_path=OUTPUT_PATH,
        save_path='output/result_full.png'
    )

    # Zoom на интересный участок (центральная часть)
    print("Генерация zoom визуализации...")
    h, w = prediction_binary.shape
    zoom_size = 300
    center_h, center_w = h // 2, w // 2

    visualize_results(
        dem_path=DEM_PATH,
        prediction_path=OUTPUT_PATH,
        save_path='output/result_zoom.png',
        zoom_box=(
            center_h - zoom_size // 2,
            center_h + zoom_size // 2,
            center_w - zoom_size // 2,
            center_w + zoom_size // 2
        )
    )

    print("\n✓ Готово!")
    print(f"\nФайлы сохранены в: {OUTPUT_PATH.parent.absolute()}")


if __name__ == "__main__":
    main()
