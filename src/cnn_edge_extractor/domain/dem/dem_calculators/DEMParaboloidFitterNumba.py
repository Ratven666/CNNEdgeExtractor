import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _fit_paraboloids_numba(dem_arr, window_size, ridge_lambda):
    """
    dem_arr      : 2D float64, может содержать NaN
    window_size  : нечётное целое
    ridge_lambda : >= 0, коэффициент L2‑регуляризации

    Возвращает: coeffs[h, w, 6] (float64)
    """
    H, W = dem_arr.shape
    k = window_size
    half = k // 2
    lam = ridge_lambda

    coeffs = np.full((H, W, 6), np.nan, dtype=np.float64)

    # единичная матрица 6x6
    eye6 = np.zeros((6, 6), dtype=np.float64)
    for d in range(6):
        eye6[d, d] = 1.0

    # предвычисленные координаты в окне k x k
    xs_full = np.empty((k, k), dtype=np.int64)
    ys_full = np.empty((k, k), dtype=np.int64)
    for yy in range(k):
        for xx in range(k):
            ys_full[yy, xx] = yy - half
            xs_full[yy, xx] = xx - half

    # параллельно по строкам
    for i in prange(H):
        # границы по y для текущей строки
        y_min = 0 if i - half < 0 else i - half
        y_max = H if i + half + 1 > H else i + half + 1

        for j in range(W):
            # границы по x
            x_min = 0 if j - half < 0 else j - half
            x_max = W if j + half + 1 > W else j + half + 1

            ky = y_max - y_min
            kx = x_max - x_min

            # буферы для валидных точек
            max_points = ky * kx
            zs_buf = np.empty(max_points, dtype=np.float64)
            xs_buf = np.empty(max_points, dtype=np.float64)
            ys_buf = np.empty(max_points, dtype=np.float64)
            n_valid = 0

            for yy in range(ky):
                for xx in range(kx):
                    z = dem_arr[y_min + yy, x_min + xx]
                    if z == z:  # проверка на not-NaN (NaN != NaN)
                        ys_buf[n_valid] = ys_full[half - (i - y_min) + yy,
                                                  half - (j - x_min) + xx]
                        xs_buf[n_valid] = xs_full[half - (i - y_min) + yy,
                                                  half - (j - x_min) + xx]
                        zs_buf[n_valid] = z
                        n_valid += 1

            if n_valid < 6:
                # недостаточно точек для оценки 6 параметров
                continue

            # формируем G (n_valid x 6)
            G = np.empty((n_valid, 6), dtype=np.float64)
            for t in range(n_valid):
                x = xs_buf[t]
                y = ys_buf[t]
                G[t, 0] = x * x
                G[t, 1] = y * y
                G[t, 2] = x * y
                G[t, 3] = x
                G[t, 4] = y
                G[t, 5] = 1.0

            # GTG = G^T G, GTz = G^T z
            GTG = np.zeros((6, 6), dtype=np.float64)
            GTz = np.zeros(6, dtype=np.float64)

            for r in range(n_valid):
                z = zs_buf[r]
                for c1 in range(6):
                    v1 = G[r, c1]
                    GTz[c1] += v1 * z
                    for c2 in range(6):
                        GTG[c1, c2] += v1 * G[r, c2]

            # ridge: (GTG + λI) a = GTz
            for d in range(6):
                GTG[d, d] += lam

            # Решаем 6x6 систему методом Гаусса с частичным выбором опорного элемента
            A = GTG
            b = GTz
            ok = True

            # прямой ход
            for col in range(6):
                # выбор максимального по модулю элемента в столбце
                pivot = col
                max_val = abs(A[col, col])
                for row in range(col + 1, 6):
                    v = abs(A[row, col])
                    if v > max_val:
                        max_val = v
                        pivot = row

                if max_val == 0.0:
                    ok = False
                    break

                # swap строк
                if pivot != col:
                    for c in range(col, 6):
                        tmp = A[col, c]
                        A[col, c] = A[pivot, c]
                        A[pivot, c] = tmp
                    tmpb = b[col]
                    b[col] = b[pivot]
                    b[pivot] = tmpb

                diag = A[col, col]
                inv_diag = 1.0 / diag

                # нормируем опорную строку
                for c in range(col, 6):
                    A[col, c] *= inv_diag
                b[col] *= inv_diag

                # зануляем ниже
                for row in range(col + 1, 6):
                    factor = A[row, col]
                    if factor != 0.0:
                        for c in range(col, 6):
                            A[row, c] -= factor * A[col, c]
                        b[row] -= factor * b[col]

            if not ok:
                continue

            # обратный ход
            x_sol = np.zeros(6, dtype=np.float64)
            for row in range(5, -1, -1):
                s = b[row]
                for c in range(row + 1, 6):
                    s -= A[row, c] * x_sol[c]
                x_sol[row] = s

            for c in range(6):
                coeffs[i, j, c] = x_sol[c]

    return coeffs


class DEMParaboloidFitterNumba:
    """
    Быстрая (numba, parallel) и устойчивaя (ridge) оценка
    параболической поверхности в каждой ячейке DEM.

    z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    """

    def __init__(self, dem, window_size: int = 3, ridge_lambda: float = 1e-6):
        if window_size % 2 == 0:
            raise ValueError("window_size должен быть нечётным")
        if dem.dem_array is None:
            raise ValueError("dem.dem_array не инициализирован")

        self.dem = dem
        self.dem_array = np.asarray(dem.dem_array, dtype=np.float64)
        self.window_size = int(window_size)
        self.ridge_lambda = float(ridge_lambda)

    def compute_coefficients(self):
        """
        Возвращает numpy.ndarray формы [H, W, 6], dtype=float64.
        Первая компиляция numba займет время; последующие вызовы быстрые.
        """
        return _fit_paraboloids_numba(
            self.dem_array,
            self.window_size,
            self.ridge_lambda,
        )


if __name__ == "__main__":
    from app.base.dem.Dem import Dem

    dem = Dem.load(file_path="../grib_05m.tif")
    cropped_dem = dem.crop(
        x_min=280500,
        y_min=535200,
        crop_width=1000,
        crop_height=1000,
    )

    pf = DEMParaboloidFitterNumba(
        dem=cropped_dem,
        window_size=9,
        ridge_lambda=1e-5,
    )
    c_map = pf.compute_coefficients()
    print(c_map.shape)
