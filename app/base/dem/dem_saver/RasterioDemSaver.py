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

        with rasterio.open(
                f"{file_path}",
                "w",
                driver="GTiff",
                height=self.dem.height,
                width=self.dem.width,
                count=1,
                dtype=self.dem.dem_array.dtype,
                # crs="EPSG:32636",  # укажи свой EPSG, например 32636 для UTM 36N
                transform=transform,
                nodata=np.nan
        ) as dst:
            dst.write(self.dem.dem_array, 1)


