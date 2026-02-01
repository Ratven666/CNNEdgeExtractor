
from app.base.mesh.parsers.MeshDxfParser import MeshDxfParser
from app.base.mesh.parsers.MeshParserABC import MeshParserABC


class MeshParserFactory(MeshParserABC):

    mesh_parsers = {"dxf": MeshDxfParser,
                    # "las": ScanParserFromLas,
                    }

    def __init__(self, mesh):
        super().__init__(mesh)
        self.parser: MeshParserABC | None = None

    def parse(self, file_path: str):
        self.parser = self.__get_mesh_parser(file_path)
        self.parser.parse(filepath=file_path)
        return self.mesh

    def __get_mesh_parser(self, file_path):
        file_extension = file_path.strip().split(".")[-1]
        parser_class = self.mesh_parsers[file_extension]
        parser_obj = parser_class(mesh=self.mesh)
        return parser_obj
