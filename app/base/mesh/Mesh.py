import trimesh
from sklearn.neighbors import NearestNeighbors

from app.base.mesh.MeshIterator import MeshIterator
from app.base.mesh.mesh_triangulators.ScipyTriangulator import ScipyTriangulator
from app.base.mesh.parsers.MeshParserFactory import MeshParserFactory


class Mesh:

    def __init__(self, name="DefaultMeshName"):
        self.name = name
        self.mesh = None
        self.vertices_colors = None
        self.face_colors = None

    def __iter__(self):
        return iter(MeshIterator(self))

    def __len__(self):
        return len(self.mesh.faces)

    def __str__(self):
        return f"Mesh: (len={len(self)}, name={self.name})"

    def get_z_by_xy(self, x, y):
        ray_origin = [x, y, self.mesh.bounds[1][2] + 10]  # z = max_z + 10
        ray_direction = [0, 0, -1]  # Направление вниз
        locations, _, _ = self.mesh.ray.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction]
        )
        if len(locations) > 0:
            z = locations[0][2]  # Берём первую точку пересечения
            return z
        else:
            return None

    def rtx_by_dirs(self, ray_origins, ray_directions):
        ray_directions = trimesh.util.unitize(ray_directions)
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions)
        return locations, index_ray, index_tri

    def create_mesh_from_scan(self, scan, scan_triangulator=ScipyTriangulator):
        scan_triangulator().create_mesh(mesh=self, scan=scan)
        return self

    def load_mesh_from_file(self, filepath, parser=MeshParserFactory):
        parser = parser(mesh=self)
        parser.parse(filepath)
        return self

    def simplify_mesh(self, face_count=None, target_reduction=None, save_colors=True):
        if face_count is not None:
            simplified = self.mesh.simplify_quadric_decimation(face_count=face_count)
        elif target_reduction is not None:
            simplified = self.mesh.simplify_quadric_decimation(percent=target_reduction)
        else:
            raise ValueError("Нет параметров для разрежения")
        if save_colors:
            nn = NearestNeighbors(n_neighbors=1).fit(self.mesh.vertices)
            _, indices = nn.kneighbors(simplified.vertices)
            # Переносим цвета
            simplified.visual.vertex_colors = self.mesh.visual.vertex_colors[indices.flatten()]
        self.mesh = simplified

    def export_mesh(self, filepath):
        self.mesh.export(filepath)

    def plot(self):
        self.mesh.show()


if __name__ == "__main__":
    mesh = Mesh().load_mesh_from_file(filepath=r"../../../src/Grib_dxf_mesh.dxf")
    print(mesh)
    print(len(mesh))
    # mesh.plot()

    mesh.simplify_mesh(face_count=1000)
    # mesh.plot()
    print(len(mesh))
    # for t in mesh:
    #     print(t)
    # mesh.export_mesh("./mesh.ply")

    bounds = mesh.mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]

    # Центр AABB (середина по осям X и Y)
    center_x = (bounds[0][0] + bounds[1][0]) / 2
    center_y = (bounds[0][1] + bounds[1][1]) / 2
    z = mesh.get_z_by_xy(center_x, center_y)
    print(center_x, center_y, z)
