import numpy as np
import rasterio
from rasterio.transform import from_bounds

from app.base.dem.dem_saver.DemSaverABC import DemSaverABC


class RasterioDemSaver(DemSaverABC):

    def __init__(self, dem):
        super().__init__(dem=dem)


    def save(self, file_path):
        transform = from_bounds(self.dem.bounds["x_min"],
                                self.dem.bounds["y_min"],
                                self.dem.bounds["x_max"],
                                self.dem.bounds["y_max"],
                                self.dem.width,
                                self.dem.height)

        dtype = self.dem.dem_array.dtype
        nodata = np.nan
        if np.issubdtype(dtype, np.integer):
            nodata = None  # или 0, если хочешь явный код

        with rasterio.open(
                file_path, "w",
                driver="GTiff",
                height=self.dem.height,
                width=self.dem.width,
                count=1,
                dtype=dtype,
                transform=transform,
                nodata=nodata,
        ) as dst:
            dst.write(self.dem.dem_array, 1)
