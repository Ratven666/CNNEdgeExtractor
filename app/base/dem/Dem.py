import numpy as np
import rasterio

from app.base.dem.dem_creators.DemCreatorFromMesh import DemCreatorFromMesh
from app.base.dem.dem_loader.RasterioDemLoader import RasterioDemLoader
from app.base.dem.dem_saver.RasterioDemSaver import RasterioDemSaver
from app.base.dem.plotters.MplDemPlotter import MplDemPlotter
from app.base.dem.utils.DemCropper import DemCropper


class Dem:

    def __init__(self, resolution, name="DefaultDemName"):
        self.name = name
        self.resolution = resolution
        self.dem_array = None
        self.width, self.height = None, None
        self.bounds = {}

    @classmethod
    def create_dem_from_mesh(cls, data_odj, resolution, name="DefaultDemName", dem_creator=DemCreatorFromMesh):
        dem = cls(name=name, resolution=resolution)
        dem_creator = dem_creator(dem=dem)
        dem = dem_creator.create(data_odj)
        return dem

    def save(self, file_path, object_saver=RasterioDemSaver):
        obj_saver = object_saver(dem=self)
        obj_saver.save(file_path)

    @classmethod
    def load(cls, file_path, name="LoadedDem", loader=RasterioDemLoader):
        dem_data = loader().load(file_path)
        dem = cls(name=name, resolution=dem_data["resolution"])
        dem.dem_array = dem_data["dem_array"]
        dem.width = dem_data["width"]
        dem.height = dem_data["height"]
        dem.bounds = dem_data["bounds"]
        return dem

    def crop(self, x_min, y_min, x_max=None, y_max=None,
             crop_width=None, crop_height=None,
             cropper=DemCropper):
        cropper = cropper(self)
        cropped_dem = cropper.crop(x_min, y_min, x_max, y_max, crop_width, crop_height)
        return cropped_dem

    def plot(self, plotter=MplDemPlotter):
        plotter = plotter(self)
        plotter.plot()

    def __len__(self):
        return self.width * self.height

    def __str__(self):
        return (f"Dem: (name={self.name}, res={self.resolution}, "
                f"WH=[{self.width}, {self.height}], len={len(self)})")


if __name__ == "__main__":
    from app.base.mesh.Mesh import Mesh

    mesh = Mesh(name="Grib").load_mesh_from_file(filepath=r"../../../src/Grib_dxf_mesh.dxf")
    print(mesh)

    dem = Dem.create_dem_from_mesh(data_odj=mesh,
                                   resolution=1,
                                   name="Grib",
                                   dem_creator=DemCreatorFromMesh)

    print(dem)

    dem.save(file_path="grib_1m.tif")

    # dem = Dem.load(file_path="dem.tif")
    #
    # # dem.plot()
    # print(dem)
    #
    # cropped_dem = dem.crop(x_min=280500, y_min=535200, crop_width=100,
    #                        crop_height=100)
    # # cropped_dem.plot()
    # cropped_dem.save(file_path="cr_1.tif")