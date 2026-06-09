from copy import deepcopy

import numpy as np

from src.cnn_edge_extractor.domain.dem.Dem import Dem
from src.cnn_edge_extractor.domain.dem.dem_calculators.DEMParaboloidFitterNumba import DEMParaboloidFitterNumba


class DEMMorphometryCalculator:

    def __init__(self, dem: Dem, window_size=3, fitter=DEMParaboloidFitterNumba):
        self.dem = dem
        self.window_size = window_size
        self._fitter = fitter(dem=dem, window_size=window_size)
        self._coefficients_map = None

    @property
    def coefficients_map(self):
        if self._coefficients_map is None:
            self._coefficients_map = self._fitter.compute_coefficients()
        return self._coefficients_map

    def compute_slope(self) -> Dem:
        # Берём d и e из коэффициентов (они и есть производные в (0,0))
        d = self.coefficients_map[..., 3]  # dz/dx в центре
        e = self.coefficients_map[..., 4]  # dz/dy в центре
        p, q = d, e
        # Модуль градиента
        grad_mag = np.sqrt(p**2 + q**2)
        # Наклон
        slope_rad = np.arctan(grad_mag)
        slope_deg = np.degrees(slope_rad)

        slope_dem = deepcopy(self.dem)
        slope_dem.dem_array = slope_deg
        return slope_dem

    def compute_aspect(self) -> Dem:
        # Берём d и e из коэффициентов (они и есть производные в (0,0))
        d = self.coefficients_map[..., 3]  # dz/dx в центре
        e = self.coefficients_map[..., 4]  # dz/dy в центре
        p, q = d, e
        # Модуль градиента
        grad_mag = np.sqrt(p**2 + q**2)
        # Аспект (0 = север, 90 = восток, по часовой, как в GIS)
        aspect_rad = np.arctan2(p, -q)
        aspect_deg = np.degrees(aspect_rad)
        aspect_deg = np.where(aspect_deg < 0, 360.0 + aspect_deg, aspect_deg)

        # Плоские участки: аспект не определён
        flat_mask = grad_mag == 0
        aspect_deg[flat_mask] = np.nan

        aspect_dem = deepcopy(self.dem)
        aspect_dem.dem_array = aspect_deg
        return aspect_dem

    def compute_total_curvature(self) -> Dem:
        a = self.coefficients_map[..., 0]
        b = self.coefficients_map[..., 1]
        c = self.coefficients_map[..., 2]
        d = self.coefficients_map[..., 3]  # p = z_x(0,0)
        e = self.coefficients_map[..., 4]  # q = z_y(0,0)
        # Вторые производные в центре (0,0)
        z_xx = 2.0 * a
        z_yy = 2.0 * b
        z_xy = c
        p, q = d, e
        # Формула средней (total/mean) кривизны:
        # K = [ (1+q^2) z_xx - 2 p q z_xy + (1+p^2) z_yy ] / [ 2 (1 + p^2 + q^2)^(3/2) ]
        denom = 2.0 * (1.0 + p ** 2 + q ** 2) ** 1.5
        # Чтобы избежать деления на ноль, можно добавить маленький эпсилон
        eps = 1e-12
        denom_safe = denom + eps
        num = (1.0 + q ** 2) * z_xx - 2.0 * p * q * z_xy + (1.0 + p ** 2) * z_yy
        total_curv = num / denom_safe
        total_curvature = deepcopy(self.dem)
        total_curvature.dem_array = total_curv
        return total_curvature

if __name__ == '__main__':


    dem = Dem.load(file_path="../grib_05m.tif")

    # dem.plot()
    print(dem)

    cropped_dem = dem.crop(x_min=280500, y_min=535200, crop_width=100,
                           crop_height=100)

    pf = DEMMorphometryCalculator(dem=dem, window_size=3,)

    slope_dem = pf.compute_slope()
    # print(type(c_map), c_map.shape)
    print(type(cropped_dem.dem_array), cropped_dem.dem_array.shape)
    slope_dem.save(file_path="../slope_grib_05m.tif")
    slope_dem.plot()

    aspect_dem = pf.compute_aspect()
    aspect_dem.save(file_path="../aspect_grib_05m.tif")
    aspect_dem.plot()

    total_curvature = pf.compute_total_curvature()
    total_curvature.save(file_path="../total_curvature_grib_05m.tif")
    total_curvature.plot()