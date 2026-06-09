from abc import ABC, abstractmethod


class DemCreatorABC(ABC):

    def __init__(self, dem):
        self.dem = dem

    @abstractmethod
    def create(self, data_odj):
        pass
