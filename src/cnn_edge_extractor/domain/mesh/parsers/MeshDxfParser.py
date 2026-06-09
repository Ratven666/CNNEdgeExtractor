import ezdxf
import numpy as np
import trimesh

from src.cnn_edge_extractor.domain.mesh.parsers.MeshParserABC import MeshParserABC


class MeshDxfParser(MeshParserABC):

    def __init__(self, mesh):
        super().__init__(mesh)

    def parse(self, filepath):
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()

        vertices = []
        faces_idx = []

        for face in msp.query("3DFACE"):
            v = [
                np.array(face.dxf.vtx0),
                np.array(face.dxf.vtx1),
                np.array(face.dxf.vtx2),
                np.array(face.dxf.vtx3),
            ]
            # треугольник (3-я и 4-я совпадают)
            if np.allclose(v[2], v[3]):
                base = len(vertices)
                vertices.extend(v[:3])
                faces_idx.append([base, base + 1, base + 2])
            else:
                base = len(vertices)
                vertices.extend(v)
                faces_idx.append([base, base + 1, base + 2])
                faces_idx.append([base, base + 2, base + 3])

        vertices = np.array(vertices)
        faces_idx = np.array(faces_idx)

        tri = trimesh.Trimesh(vertices=vertices, faces=faces_idx, process=False)
        self.mesh.mesh = tri
        return self.mesh

if __name__ == "__main__":
    from app.base.mesh.Mesh import Mesh
    mesh = Mesh("DXF mesh")
    mesh = MeshDxfParser(mesh).parse(filepath=r"../../../../src/Grib_dxf_mesh.dxf")
    print(mesh, len(mesh))
    mesh.plot()
