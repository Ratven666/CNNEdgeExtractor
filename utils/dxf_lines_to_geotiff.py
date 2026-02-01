# import ezdxf
# import numpy as np
# import rasterio
# from rasterio.features import rasterize
# from rasterio.transform import from_bounds
# from shapely.geometry import LineString
#
# def dxf_lines_to_geotiff(source_file_path, target_file_path, resolution=1.0):
#     # 1. Читаем полилинии из DXF
#     doc = ezdxf.readfile(source_file_path)
#     msp = doc.modelspace()
#     # 2. Конвертируем в Shapely LineString объекты
#     lines = []
#     for lwpoly in msp.query("LWPOLYLINE"):
#         points = np.array(lwpoly.get_points(format="xy"))
#         lines.append(LineString(points))
#     # 3. Определяем bounds для растра
#     all_coords = np.vstack([np.array(line.coords) for line in lines])
#     xmin, ymin = all_coords.min(axis=0)
#     xmax, ymax = all_coords.max(axis=0)
#
#     # 4. Задаём разрешение
#     width = int((xmax - xmin) / resolution)
#     height = int((ymax - ymin) / resolution)
#
#     # 5. Создаём affine transform
#     transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
#
#     # 6. Растеризуем линии (значение 1 для линий, 0 для фона)
#     raster = rasterize(
#         [(line, 1) for line in lines],  # (geometry, value) пары
#         out_shape=(height, width),
#         transform=transform,
#         fill=0,           # фоновое значение
#         all_touched=True, # все пиксели, которых касается линия
#         dtype=np.uint8
#     )
#     # 7. Сохраняем как GeoTIFF
#     with rasterio.open(
#         target_file_path,
#         "w",
#         driver="GTiff",
#         height=height,
#         width=width,
#         count=1,
#         dtype=raster.dtype,
#         transform=transform,
#         nodata=0
#     ) as dst:
#         dst.write(raster, 1)
#
#     print(f"Растр создан: {width}x{height} пикселей")
#
# ######################################################################################
#
# RESOLUTION = 1.0
#
# for idx in range(1, 11):
#     dxf_lines_to_geotiff(source_file_path=f"../src/filter_squares/f_square_{idx}.dxf",
#                          target_file_path=f"../src/filter_squares/square_{idx}_res_{RESOLUTION}.tif",
#                          resolution=RESOLUTION)

import ezdxf
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import LineString


def dxf_lines_to_geotiff(source_file_path, target_file_path, resolution=1.0):
    # 1. Читаем полилинии из DXF
    doc = ezdxf.readfile(source_file_path)
    msp = doc.modelspace()

    # 2. Конвертируем в Shapely LineString объекты
    lines = []
    for lwpoly in msp.query("LWPOLYLINE"):
        points = np.array(lwpoly.get_points(format="xy"))
        lines.append(LineString(points))

    # 3. Определяем исходные bounds
    all_coords = np.vstack([np.array(line.coords) for line in lines])
    xmin_raw, ymin_raw = all_coords.min(axis=0)
    xmax_raw, ymax_raw = all_coords.max(axis=0)

    # 4. Округляем границы до кратных разрешению
    xmin = np.floor(xmin_raw / resolution) * resolution
    ymin = np.floor(ymin_raw / resolution) * resolution
    xmax = np.ceil(xmax_raw / resolution) * resolution
    ymax = np.ceil(ymax_raw / resolution) * resolution

    # 5. Вычисляем размеры растра (теперь точно целые числа)
    width = int(np.round((xmax - xmin) / resolution))
    height = int(np.round((ymax - ymin) / resolution))

    # 6. Проверка: границы должны быть точно кратны разрешению
    assert np.isclose((xmax - xmin) / resolution, width), "Width mismatch!"
    assert np.isclose((ymax - ymin) / resolution, height), "Height mismatch!"

    print(f"Исходные bounds: X=[{xmin_raw:.2f}, {xmax_raw:.2f}], Y=[{ymin_raw:.2f}, {ymax_raw:.2f}]")
    print(f"Округлённые:     X=[{xmin:.2f}, {xmax:.2f}], Y=[{ymin:.2f}, {ymax:.2f}]")
    print(f"Размер растра:   {width}x{height} пикселей")
    print(f"Разрешение:      {resolution} м/пиксель")
    print(f"Проверка:        ({xmax - xmin}) / {width} = {(xmax - xmin) / width:.6f}")

    # 7. Создаём affine transform
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    # 8. Растеризуем линии
    raster = rasterize(
        [(line, 1) for line in lines],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        # dtype=np.uint8
        dtype=np.float32
    )

    # 9. Сохраняем как GeoTIFF
    with rasterio.open(
            target_file_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=raster.dtype,
            transform=transform,
            nodata=0
    ) as dst:
        dst.write(raster, 1)

    print(f"✓ Сохранено: {target_file_path}\n")


######################################################################################

RESOLUTION = 1.0

for idx in range(1, 11):
    dxf_lines_to_geotiff(source_file_path=f"../src/filter_squares/f_square_{idx}.dxf",
                         target_file_path=f"../src/filter_squares/square_{idx}_res_{RESOLUTION}.tif",
                         resolution=RESOLUTION)
