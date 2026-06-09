import numpy as np
from tqdm import tqdm


class DEMParaboloidFitter:
    """
    Оценка параметров параболической поверхности в каждой ячейке DEM
    с использованием скользящего окна.

    Поверхность: z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f

    Результат: 3D-массив coeffs[h, w, 6] (a,b,c,d,e,f для каждой ячейки).
    """

    def __init__(self, dem, window_size=3):
        """
        dem         : экземпляр Dem (из Dem.py), у которого заполнен dem.dem_array (2D numpy)
        window_size : нечётный размер окна (3,5,7,...)
        """
        if window_size % 2 == 0:
            raise ValueError("window_size должен быть нечётным")

        self.dem = dem
        self.window_size = window_size

        if self.dem.dem_array is None:
            raise ValueError("dem.dem_array не инициализирован")

        if not isinstance(self.dem.dem_array, np.ndarray):
            self.dem.dem_array = np.array(self.dem.dem_array)

        self.height, self.width = self.dem.dem_array.shape

    def _fit_window(self, z_window):
        """
        Оценка коэффициентов по одному окну.

        z_window: 2D массив размера window_size x window_size
        Возвращает: np.array([a,b,c,d,e,f])
        """
        k = self.window_size
        half = k // 2

        # Координаты внутри окна: центр в (0,0)
        ys, xs = np.mgrid[-half:half+1, -half:half+1]
        xs = xs.ravel()
        ys = ys.ravel()
        zs = z_window.ravel()

        # Модель: z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
        G = np.column_stack([
            xs**2,
            ys**2,
            xs * ys,
            xs,
            ys,
            np.ones_like(xs)
        ])

        # Решаем переопределённую систему МНК
        # coefficients = (G^T G)^(-1) G^T z
        coefficients, *_ = np.linalg.lstsq(G, zs, rcond=None)
        return coefficients  # [a,b,c,d,e,f]

    def compute_coefficients(self, use_tqdm: bool = True):
        """
        Возвращает 3D массив коэффициентов [H, W, 6].
        Прогресс показывает обработку каждой ячейки (H*W).
        """
        H, W = self.height, self.width
        k = self.window_size
        half = k // 2

        coefficients_map = np.zeros((H, W, 6), dtype=float)

        total = H * W
        if use_tqdm:
            pbar = tqdm(total=total, desc="Fitting paraboloids", unit="cell")
        else:
            pbar = None

        for i in range(H):
            # Границы окна по y
            y_min = max(0, i - half)
            y_max = min(H, i + half + 1)

            for j in range(W):
                # Границы окна по x
                x_min = max(0, j - half)
                x_max = min(W, j + half + 1)

                z_window = self.dem.dem_array[y_min:y_max, x_min:x_max]

                mask = ~np.isnan(z_window)
                if np.count_nonzero(mask) < 6:
                    coefficients = np.full(6, np.nan)
                else:
                    k_y, k_x = z_window.shape
                    ys_full, xs_full = np.mgrid[
                        -(i - y_min):k_y - (i - y_min),
                        -(j - x_min):k_x - (j - x_min),
                    ]
                    xs_full = xs_full[mask]
                    ys_full = ys_full[mask]
                    zs = z_window[mask].ravel()

                    G = np.column_stack([
                        xs_full ** 2,
                        ys_full ** 2,
                        xs_full * ys_full,
                        xs_full,
                        ys_full,
                        np.ones_like(xs_full),
                    ])
                    try:
                        coefficients, *_ = np.linalg.lstsq(G, zs, rcond=None)
                    except np.linalg.LinAlgError:
                        coefficients = np.full(6, np.nan)

                coefficients_map[i, j, :] = coefficients

                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        return coefficients_map

if __name__ == "__main__":
    from app.base.dem.Dem import Dem

    dem = Dem.load(file_path="../grib_05m.tif")

    # dem.plot()
    print(dem)

    cropped_dem = dem.crop(x_min=280500, y_min=535200, crop_width=100,
                           crop_height=100)

    pf = DEMParaboloidFitter(dem=cropped_dem, window_size=3)

    c_map = pf.compute_coefficients()
    print(type(c_map), c_map.shape)
    print(type(cropped_dem.dem_array), cropped_dem.dem_array.shape)

# import numpy as np
# from tqdm import tqdm
# from numba import njit, prange
#
#
# @njit(parallel=True)
# def _compute_coefficients_numba(dem_arr, window_size):
#     """
#     Ядро: параллельный расчёт коэффициентов для всего DEM.
#
#     dem_arr     : 2D float64 с NaN
#     window_size : нечётное целое
#
#     Возвращает coeffs[h, w, 6] (float64), NaN где оценки нет.
#     """
#     H, W = dem_arr.shape
#     k = window_size
#     half = k // 2
#
#     coeffs = np.full((H, W, 6), np.nan, dtype=np.float64)
#
#     # предвычисляем координаты окна k x k относительно центра (0,0)
#     xs_full = np.empty((k, k), dtype=np.int64)
#     ys_full = np.empty((k, k), dtype=np.int64)
#     for yy in range(k):
#         for xx in range(k):
#             ys_full[yy, xx] = yy - half
#             xs_full[yy, xx] = xx - half
#
#     # параллелим по строкам
#     for i in prange(H):
#         # границы окна по y
#         y_min = 0 if i - half < 0 else i - half
#         y_max = H if i + half + 1 > H else i + half + 1
#
#         for j in range(W):
#             # границы окна по x
#             x_min = 0 if j - half < 0 else j - half
#             x_max = W if j + half + 1 > W else j + half + 1
#
#             ky = y_max - y_min
#             kx = x_max - x_min
#
#             # буферы под валидные точки
#             max_points = ky * kx
#             xs_buf = np.empty(max_points, dtype=np.float64)
#             ys_buf = np.empty(max_points, dtype=np.float64)
#             zs_buf = np.empty(max_points, dtype=np.float64)
#             n_valid = 0
#
#             # собираем окно и маску not-NaN
#             for yy in range(ky):
#                 for xx in range(kx):
#                     z = dem_arr[y_min + yy, x_min + xx]
#                     if z == z:  # z is not NaN
#                         ys_buf[n_valid] = ys_full[half - (i - y_min) + yy,
#                                                   half - (j - x_min) + xx]
#                         xs_buf[n_valid] = xs_full[half - (i - y_min) + yy,
#                                                   half - (j - x_min) + xx]
#                         zs_buf[n_valid] = z
#                         n_valid += 1
#
#             if n_valid < 6:
#                 # недостаточно точек для оценки шести параметров
#                 continue
#
#             # строим G (n_valid x 6)
#             G = np.empty((n_valid, 6), dtype=np.float64)
#             for t in range(n_valid):
#                 x = xs_buf[t]
#                 y = ys_buf[t]
#                 G[t, 0] = x * x
#                 G[t, 1] = y * y
#                 G[t, 2] = x * y
#                 G[t, 3] = x
#                 G[t, 4] = y
#                 G[t, 5] = 1.0
#
#             # решаем МНК через нормальные уравнения:
#             # (G^T G) a = G^T z
#             GTG = np.zeros((6, 6), dtype=np.float64)
#             GTz = np.zeros(6, dtype=np.float64)
#
#             for r in range(n_valid):
#                 z = zs_buf[r]
#                 for c1 in range(6):
#                     v1 = G[r, c1]
#                     GTz[c1] += v1 * z
#                     for c2 in range(6):
#                         GTG[c1, c2] += v1 * G[r, c2]
#
#             # Решаем 6x6 систему методом Гаусса с частичным pivot'ом
#             A = GTG
#             b = GTz
#             ok = True
#
#             # прямой ход
#             for col in range(6):
#                 pivot = col
#                 max_val = abs(A[col, col])
#                 for row in range(col + 1, 6):
#                     v = abs(A[row, col])
#                     if v > max_val:
#                         max_val = v
#                         pivot = row
#
#                 if max_val == 0.0:
#                     ok = False
#                     break
#
#                 if pivot != col:
#                     for c in range(col, 6):
#                         tmp = A[col, c]
#                         A[col, c] = A[pivot, c]
#                         A[pivot, c] = tmp
#                     tmpb = b[col]
#                     b[col] = b[pivot]
#                     b[pivot] = tmpb
#
#                 diag = A[col, col]
#                 inv_diag = 1.0 / diag
#
#                 for c in range(col, 6):
#                     A[col, c] *= inv_diag
#                 b[col] *= inv_diag
#
#                 for row in range(col + 1, 6):
#                     factor = A[row, col]
#                     if factor != 0.0:
#                         for c in range(col, 6):
#                             A[row, c] -= factor * A[col, c]
#                         b[row] -= factor * b[col]
#
#             if not ok:
#                 continue
#
#             x_sol = np.zeros(6, dtype=np.float64)
#             for row in range(5, -1, -1):
#                 s = b[row]
#                 for c in range(row + 1, 6):
#                     s -= A[row, c] * x_sol[c]
#                 x_sol[row] = s
#
#             for c in range(6):
#                 coeffs[i, j, c] = x_sol[c]
#
#     return coeffs
#
#
# class DEMParaboloidFitter:
#     """
#     Оценка параметров параболической поверхности в каждой ячейке DEM
#     с использованием скользящего окна, ускоренная numba.
#     """
#
#     def __init__(self, dem, window_size=3):
#         if window_size % 2 == 0:
#             raise ValueError("window_size должен быть нечётным")
#
#         self.dem = dem
#         self.window_size = int(window_size)
#
#         if self.dem.dem_array is None:
#             raise ValueError("dem.dem_array не инициализирован")
#
#         if not isinstance(self.dem.dem_array, np.ndarray):
#             self.dem.dem_array = np.array(self.dem.dem_array, dtype=float)
#         else:
#             if self.dem.dem_array.dtype != np.float64:
#                 self.dem.dem_array = self.dem.dem_array.astype(np.float64)
#
#         self.height, self.width = self.dem.dem_array.shape
#
#     def compute_coefficients(self, use_tqdm: bool = True):
#         """
#         Возвращает 3D массив коэффициентов [H, W, 6].
#         Прогрессбар показывает обработку строк (для numba ядра прогресс по ячейкам не трекаем).
#         """
#         # numba‑ядро считает всё за один вызов; чтобы сохранить прогресс,
#         # можно перед первой «боевой» прогонкой дать маленький прогон для компиляции.
#         # Но обычно достаточно одного вызова без tqdm.
#         if use_tqdm:
#             # простой прогрессбар на уровне «один вызов»
#             with tqdm(total=1, desc="Fitting paraboloids (numba)", unit="run") as pbar:
#                 coeffs = _compute_coefficients_numba(self.dem.dem_array, self.window_size)
#                 pbar.update(1)
#         else:
#             coeffs = _compute_coefficients_numba(self.dem.dem_array, self.window_size)
#
#         return coeffs
#
#
# if __name__ == "__main__":
#     from app.base.dem.Dem import Dem
#
#     dem = Dem.load(file_path="../grib_05m.tif")
#     print(dem)
#
#     cropped_dem = dem.crop(
#         x_min=280500,
#         y_min=535200,
#         crop_width=100,
#         crop_height=100,
#     )
#
#     pf = DEMParaboloidFitter(dem=cropped_dem, window_size=3)
#     c_map = pf.compute_coefficients()
#     print(type(c_map), c_map.shape)
#     print(type(cropped_dem.dem_array), cropped_dem.dem_array.shape)

