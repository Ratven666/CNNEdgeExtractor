from abc import ABC, abstractmethod


class DemSaverABC(ABC):

    def __init__(self, dem):
        self.dem = dem

    @abstractmethod
    def save(self, file_path):
        pass
