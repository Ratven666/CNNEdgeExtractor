from src.cnn_edge_extractor.domain.Triangle import Triangle
from src.cnn_edge_extractor.domain.scan.ScanPoint import ScanPoint


class MeshIterator:

    def __init__(self, mesh):
        self.mesh = mesh
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < len(self.mesh):
            face = self.mesh.mesh.faces[self.idx]
            points = []
            for point_idx in face:
                point = ScanPoint(x=self.mesh.mesh.vertices[point_idx][0],
                                  y=self.mesh.mesh.vertices[point_idx][1],
                                  z=self.mesh.mesh.vertices[point_idx][2],
                                  color=self.mesh.mesh.visual.vertex_colors[point_idx][:3])
                points.append(point)
            triangle = Triangle(*points)
            self.idx += 1
            return triangle
        raise StopIteration

