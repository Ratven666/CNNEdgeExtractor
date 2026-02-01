from matplotlib import pyplot as plt

from app.base.dem.plotters.DemPlotterABC import DemPlotterABC


class MplDemPlotter(DemPlotterABC):

    def __init__(self, dem):
        super(MplDemPlotter, self).__init__(dem)

    def plot(self):
        plt.figure(figsize=(12, 8))
        plt.imshow(self.dem.dem_array, cmap='terrain', origin='upper')
        plt.colorbar(label='Высота, м')
        plt.title('DEM карьера')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        plt.show()
