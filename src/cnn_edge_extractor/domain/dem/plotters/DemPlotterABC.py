from abc import ABC


class DemPlotterABC(ABC):

    def __init__(self, dem):
        self.dem = dem

    def plot(self, *args, **kwargs):
        pass
