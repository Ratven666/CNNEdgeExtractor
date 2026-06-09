import numpy as np
from tqdm import tqdm

from src.cnn_edge_extractor.domain.dem.dem_creators.DemCreatorABC import DemCreatorABC


class DemCreatorFromMesh(DemCreatorABC):

    def __init__(self, dem):
        super().__init__(dem)
        self.mesh = None

    def __init_bounds(self):
        bounds = self.mesh.mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        x_min, y_min = bounds[0][0], bounds[0][1]
        x_max, y_max = bounds[1][0], bounds[1][1]

        # Округляем границы до кратных разрешению
        x_min = np.floor(x_min / self.dem.resolution) * self.dem.resolution
        y_min = np.floor(y_min / self.dem.resolution) * self.dem.resolution
        x_max = np.ceil(x_max / self.dem.resolution) * self.dem.resolution
        y_max = np.ceil(y_max / self.dem.resolution) * self.dem.resolution

        return {"x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                }


    def create(self, data_odj):
        self.mesh = data_odj
        self.dem.bounds = self.__init_bounds()
        self.dem.width = int((self.dem.bounds["x_max"] - self.dem.bounds["x_min"]) / self.dem.resolution)
        self.dem.height = int((self.dem.bounds["y_max"] - self.dem.bounds["y_min"]) / self.dem.resolution)

        dem_array = np.full((self.dem.height, self.dem.width), np.nan, dtype=np.float32)

        total_pixels = self.dem.height * self.dem.width
        with tqdm(total=total_pixels, desc="Создание DEM", unit="px") as pbar:
            for i in range(self.dem.height):
                for j in range(self.dem.width):
                    x = self.dem.bounds["x_min"] + j * self.dem.resolution + self.dem.resolution / 2
                    y = self.dem.bounds["y_max"] - i * self.dem.resolution - self.dem.resolution / 2  # Y идёт сверху вниз
                    z = self.mesh.get_z_by_xy(x, y)
                    if z is not None:
                        dem_array[i, j] = z
                    pbar.update(1)

        self.dem.dem_array = dem_array
        return self.dem


if __name__ == "__main__":
    from app.base.mesh.Mesh import Mesh
    from app.base.dem.Dem import Dem

    mesh = Mesh(name="Grib").load_mesh_from_file(filepath=r"../../../../src/Grib_dxf_mesh.dxf")
    print(mesh)

    dem = Dem.create_from_mesh(data_odj=mesh,
                               resolution=10,
                               name="Grib",
                               dem_creator=DemCreatorFromMesh)

    print(dem)
