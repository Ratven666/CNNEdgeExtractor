from pathlib import Path

import numpy as np

from app.base.dem.Dem import Dem
from app.base.mesh.Mesh import Mesh
from app.base.scan.Scan import Scan
from app.cnn_pytorch.DEMPredictor import visualize_results, DEMPredictor

scan = Scan("PCLD")
scan.import_points_from_file("src/PCLD.las")
print(scan)

mesh = Mesh("MeshCloud")
mesh.create_mesh_from_scan(scan)
print(mesh)

dem = Dem.create_from_mesh(data_odj=mesh, resolution=0.25, name="DemCloud")
dem.save("src/PCLD_dem_025.tif")


def main():
    # Параметры
    MODEL_PATH = Path("./app/cnn_pytorch/models/best_edge_extractor.pth")
    DEM_PATH = Path("src/PCLD_dem_025.tif")  # Путь к полному DEM
    OUTPUT_PATH = Path(f"src/PCLD_Dem_predict_025m.tif")

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


main()
