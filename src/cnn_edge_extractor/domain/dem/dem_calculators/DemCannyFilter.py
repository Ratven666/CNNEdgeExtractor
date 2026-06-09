from copy import deepcopy

import numpy as np
from skimage.feature import canny

from src.cnn_edge_extractor.domain.dem.Dem import Dem


class DemCannyFilter:
    """
    Расчёт карты границ (Канни) для DEM.

    Использует skimage.feature.canny над dem.dem_array.
    """

    def __init__(self, dem: Dem):
        """
        dem : экземпляр Dem (из Dem.py), у которого заполнен dem.dem_array.
        """
        if dem.dem_array is None:
            raise ValueError("dem.dem_array не инициализирован")
        self.dem = dem

    def compute_edges(self,
                      sigma: float = 1.,
                      low_threshold=None,
                      high_threshold=None,
                      use_quantiles: bool = False) -> Dem:
        """
        Возвращает 2D-массив edges[h, w] (bool), где True = граница.

        Параметры передаются в skimage.feature.canny.[web:126][web:129]
        """
        img = np.asarray(self.dem.dem_array, dtype=float)
        # NaN убираем: можно либо заполнить интерполяцией,
        # либо просто заменить на среднее/минимум.
        nan_mask = ~np.isfinite(img)
        if nan_mask.any():
            # заполняем NaN медианой по маске
            valid_vals = img[~nan_mask]
            if valid_vals.size == 0:
                raise ValueError("В dem.dem_array нет валидных значений")
            fill_val = np.median(valid_vals)
            img = img.copy()
            img[nan_mask] = fill_val

        edges = canny(
            image=img,
            sigma=sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            use_quantiles=use_quantiles,
        )  # bool-массив[web:120][web:126]
        edges = edges.astype("uint8")  # 0/1
        canny_dem = deepcopy(self.dem)
        canny_dem.dem_array = edges
        return canny_dem


if __name__ == "__main__":
    from app.base.dem.Dem import Dem

    dem = Dem.load(file_path="../slope_grib_05m.tif")
    print(dem)

    canny_filter = DemCannyFilter(dem)
    edges = canny_filter.compute_edges(sigma=1.0,
                                       use_quantiles=True)
    edges.save(file_path="../canny_grib_05m.tif")
    edges.plot()


