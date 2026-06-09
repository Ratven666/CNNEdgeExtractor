from abc import ABC, abstractmethod



class MeshParserABC(ABC):

    def __init__(self, mesh):
        self.mesh = mesh

    @abstractmethod
    def parse(self, filepath: str):
        pass
