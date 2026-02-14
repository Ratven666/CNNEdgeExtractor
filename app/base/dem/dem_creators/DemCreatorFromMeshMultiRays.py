import numpy as np
from tqdm import tqdm

from app.base.dem.dem_creators.DemCreatorABC import DemCreatorABC


class DemCreatorFromMeshMultiRays(DemCreatorABC):
    def __init__(self, dem):
        super().__init__(dem)
        self.mesh = None

    def __init_bounds(self):
        bounds = self.mesh.mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        x_min, y_min = bounds[0][0], bounds[0][1]
        x_max, y_max = bounds[1][0], bounds[1][1]

        res = self.dem.resolution
        x_min = np.floor(x_min / res) * res
        y_min = np.floor(y_min / res) * res
        x_max = np.ceil(x_max / res) * res
        y_max = np.ceil(y_max / res) * res

        return {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        }

    def create(self, data_odj, batch_size: int = 100_000):
        """
        Быстрое создание DEM:
        - все пиксели → массив координат центров;
        - одним (или несколькими) вызовами ray.intersects_location
          на большие батчи лучей вниз.
        """
        self.mesh = data_odj
        self.dem.bounds = self.__init_bounds()

        res = self.dem.resolution
        self.dem.width = int((self.dem.bounds["x_max"] - self.dem.bounds["x_min"]) / res)
        self.dem.height = int((self.dem.bounds["y_max"] - self.dem.bounds["y_min"]) / res)

        H, W = self.dem.height, self.dem.width
        dem_array = np.full((H, W), np.nan, dtype=np.float32)

        # координаты центров всех ячеек
        xs = self.dem.bounds["x_min"] + (np.arange(W) + 0.5) * res
        ys = self.dem.bounds["y_max"] - (np.arange(H) + 0.5) * res  # сверху вниз

        X, Y = np.meshgrid(xs, ys)  # shape (H, W)

        centers = np.column_stack([X.ravel(), Y.ravel()])
        n_rays = centers.shape[0]

        # все лучи вниз, z чуть выше max_z
        max_z = self.mesh.mesh.bounds[1][2]
        ray_origins = np.column_stack(
            [centers[:, 0], centers[:, 1], np.full(n_rays, max_z + 10.0, dtype=float)]
        )
        ray_directions = np.tile(np.array([0.0, 0.0, -1.0], dtype=float), (n_rays, 1))

        # батчами пускаем лучи
        z_vals = np.full(n_rays, np.nan, dtype=np.float32)

        from app.base.mesh.Mesh import Mesh  # чтобы использовать rtx_by_dirs

        mesh_wrapper: Mesh = self.mesh

        with tqdm(total=n_rays, desc="Создание DEM (лучи)", unit="ray") as pbar:
            for start in range(0, n_rays, batch_size):
                end = min(start + batch_size, n_rays)
                origins_batch = ray_origins[start:end]
                dirs_batch = ray_directions[start:end]

                locations, index_ray, _ = mesh_wrapper.rtx_by_dirs(
                    ray_origins=origins_batch,
                    ray_directions=dirs_batch,
                )
                # locations: (M, 3), index_ray: длина M — индексы в батче

                if locations.shape[0] > 0:
                    # для каждого луча берем первую точку пересечения
                    # создаём массив z по индексу луча в батче
                    # (trimesh сортирует по расстоянию, так что первая — ближайшая)[web:21]
                    # заполняем
                    for loc, idx_local in zip(locations, index_ray):
                        z_vals[start + idx_local] = loc[2]

                pbar.update(end - start)

        dem_array[:, :] = z_vals.reshape(H, W)
        self.dem.dem_array = dem_array
        return self.dem
