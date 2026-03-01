import numpy as np


class DemCropper:

    def __init__(self, source_dem):


        self.source_dem = source_dem

    def crop(self, x_min, y_min, x_max=None, y_max=None,
             crop_width=None, crop_height=None):
        """
        Вырезает фрагмент DEM с сохранением координатной привязки.
        """

        # Режим 2: вырезаем по центру и размеру
        if x_max is None and y_max is None and crop_width is not None and crop_height is not None:
            x_max = x_min + crop_width
            y_max = y_min + crop_height

        # Проверяем, что границы заданы
        if x_min is None or y_min is None or x_max is None or y_max is None:
            raise ValueError(
                "Укажите границы (xmin, ymin, xmax, ymax) или (center_x, center_y, crop_width, crop_height)")

        # Округляем границы до кратных разрешению
        xmin_crop = np.floor(x_min / self.source_dem.resolution) * self.source_dem.resolution
        ymin_crop = np.floor(y_min / self.source_dem.resolution) * self.source_dem.resolution
        xmax_crop = np.ceil(x_max / self.source_dem.resolution) * self.source_dem.resolution
        ymax_crop = np.ceil(y_max / self.source_dem.resolution) * self.source_dem.resolution

        # Ограничиваем границами исходного DEM
        xmin_crop = max(xmin_crop, self.source_dem.bounds["x_min"])
        ymin_crop = max(ymin_crop, self.source_dem.bounds["y_min"])
        xmax_crop = min(xmax_crop, self.source_dem.bounds["x_max"])
        ymax_crop = min(ymax_crop, self.source_dem.bounds["y_max"])

        # Преобразуем мировые координаты в индексы пикселей
        col_min = int(np.round((xmin_crop - self.source_dem.bounds["x_min"]) / self.source_dem.resolution))
        col_max = int(np.round((xmax_crop - self.source_dem.bounds["x_min"]) / self.source_dem.resolution))
        row_min = int(np.round((self.source_dem.bounds["y_max"] - ymax_crop) / self.source_dem.resolution))
        row_max = int(np.round((self.source_dem.bounds["y_max"] - ymin_crop) / self.source_dem.resolution))

        # Ограничиваем индексы
        col_min = max(0, col_min)
        col_max = min(self.source_dem.width, col_max)
        row_min = max(0, row_min)
        row_max = min(self.source_dem.height, row_max)

        # Вырезаем фрагмент
        cropped_array = self.source_dem.dem_array[row_min:row_max, col_min:col_max].copy()

        # Создаём новый объект Dem
        from app.base.dem.Dem import Dem
        cropped_dem = Dem(resolution=self.source_dem.resolution, name=f"{self.source_dem.name}_cropped")
        cropped_dem.dem_array = cropped_array
        cropped_dem.height, cropped_dem.width = cropped_array.shape
        cropped_dem.bounds = {
            "x_min": xmin_crop,
            "y_min": ymin_crop,
            "x_max": xmax_crop,
            "y_max": ymax_crop
        }

        print(f"Вырезан фрагмент: {cropped_dem.width}x{cropped_dem.height} пикселей")
        print(f"Новые bounds: X=[{xmin_crop:.2f}, {xmax_crop:.2f}], Y=[{ymin_crop:.2f}, {ymax_crop:.2f}]")

        return cropped_dem
