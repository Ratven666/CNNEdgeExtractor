import numpy as np
import trimesh
from scipy.spatial import Delaunay


class ScipyTriangulator:

    def __init__(self):
        self.scan = None
        self.vertices = None
        self.vertices_colors = None
        self.faces = None
        self.face_colors = None

    def __str__(self):
        return (f"{self.__class__.__name__} "
                f"[Name: {self.scan.scan_name}\t\t"
                f"Count_of_triangles: {len(self.faces)}]"
                )

    def __get_data_dict(self):
        """
        Возвращает словарь с данными для построения триангуляции
        :return: словарь с данными для построения триангуляции
        """
        x_lst, y_lst, z_lst, c_lst = [], [], [], []
        for point in self.scan:
            x_lst.append(point.x)
            y_lst.append(point.y)
            z_lst.append(point.z)
            c_lst.append(point.color)
        return {"x": x_lst, "y": y_lst, "z": z_lst, "color": c_lst}

    def __calk_delone_triangulation(self):
        """
        Рассчитываает треугольники между точками
        :return: славарь с указанием вершин треугольников
        """
        points2D = self.vertices[:, :2]
        tri = Delaunay(points2D)
        i_lst, j_lst, k_lst = ([triplet[c] for triplet in tri.simplices] for c in range(3))
        return {"i_lst": i_lst, "j_lst": j_lst, "k_lst": k_lst, "ijk": tri.simplices}

    @staticmethod
    def __calk_faces_colors(ijk_dict, scan_data):
        """
        Рассчитывает цвета треугольников на основании усреднения цветов точек, образующих
        треугольник
        :param ijk_dict: словарь с вершинами треугольников
        :param scan_data: словарь с данными о точках скана
        :return: список цветов треугольников в формате [r, g, b] от 0 до 255
        """
        c_lst = []
        for idx in range(len(ijk_dict["i_lst"])):
            c_i = scan_data["color"][ijk_dict["i_lst"][idx]]
            c_j = scan_data["color"][ijk_dict["j_lst"][idx]]
            c_k = scan_data["color"][ijk_dict["k_lst"][idx]]
            r = round((c_i[0] + c_j[0] + c_k[0]) / 3)
            g = round((c_i[1] + c_j[1] + c_k[1]) / 3)
            b = round((c_i[2] + c_j[2] + c_k[2]) / 3)
            # c_lst.append([r, g, b])
            c_lst.append([r, g, b, 255])
        return c_lst

    def _triangulate(self, mesh):
        scan_data = self.__get_data_dict()
        self.vertices = np.vstack([scan_data["x"], scan_data["y"], scan_data["z"]]).T
        self.vertices_colors = scan_data["color"]
        tri_data_dict = self.__calk_delone_triangulation()
        self.faces = tri_data_dict["ijk"]
        self.face_colors = self.__calk_faces_colors(tri_data_dict, scan_data)
        return self

    def create_mesh(self, mesh, scan):
        self.scan = scan
        self._triangulate(mesh)
        mesh.mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        mesh.mesh.visual.face_colors = self.face_colors
        mesh.mesh.visual.vertex_colors = self.vertices_colors
        mesh.vertices_colors = self.vertices_colors
        mesh.face_colors = self.face_colors
        return mesh
