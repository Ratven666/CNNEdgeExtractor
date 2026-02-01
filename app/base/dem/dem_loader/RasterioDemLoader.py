import rasterio

from app.base.dem.dem_loader.DemLoaderABC import DemLoaderABC


class RasterioDemLoader(DemLoaderABC):

    def load(self, file_path):
        """Загружает DEM из GeoTIFF файла"""
        with rasterio.open(file_path) as src:
            dem_array = src.read(1)  # Читаем первый канал
            transform = src.transform
            width = src.width
            height = src.height

            # Вычисляем bounds из transform
            x_min = transform.c
            y_max = transform.f
            x_max = x_min + width * transform.a
            y_min = y_max + height * transform.e

            # Вычисляем разрешение
            resolution = abs(transform.a)

            dem_data = {"bounds": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            },
                "dem_array": dem_array,
                "resolution": resolution,
                "width": width,
                "height": height,
            }

            return dem_data
