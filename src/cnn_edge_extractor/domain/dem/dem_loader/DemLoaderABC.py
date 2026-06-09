from abc import ABC, abstractmethod


class DemLoaderABC(ABC):

    @abstractmethod
    def load(self, file_path):
        pass
