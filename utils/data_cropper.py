import os

from app.base.dem.Dem import Dem

BASE_DEM_PATH = "../src/grib_1m.tif"
LINES_DIR = "../src/tif_sq_1m"

coordinates = {1: (280500, 535200),
               2: (280300, 535200),
               3: (280500, 535400),
               4: (280320, 535390),
               5: (280300, 535500),
               6: (280200, 535400),
               8: (280100, 535600),
               9: (280000, 535000),
               10: (280000, 535400),
               }


def crop_data(coordinates,
              crop_width, crop_height,
              base_bem_path=BASE_DEM_PATH,
              lines_path=LINES_DIR,
              ):
    base_dem = Dem.load(file_path=base_bem_path)
    for idx, (x, y) in coordinates.items():
        line_path = os.path.join(lines_path, f"square_{idx}_res_1.0.tif")
        line_dem = Dem.load(file_path=line_path)
        cropped_base_dem = base_dem.crop(x_min=x, y_min=y, crop_width=crop_width, crop_height=crop_height)
        cropped_line_dem = line_dem.crop(x_min=x, y_min=y, crop_width=crop_width, crop_height=crop_height)
        cropped_base_dem.save(f"bd_res_1m_{idx}.tif")
        cropped_line_dem.save(f"line_res_1m_{idx}.tif")
        print(idx, x, y, line_path)


crop_data(coordinates, 100, 100)
