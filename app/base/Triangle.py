from app.base.Point import Point


class Triangle:

    __slots__ = ["id", "point_0", "point_1", "point_2"]

    def __init__(self, point_0: Point, point_1: Point, point_2: Point):
        self.point_0 = point_0
        self.point_1 = point_1
        self.point_2 = point_2

    def __str__(self):
        return f"{self.__class__.__name__}=[{self.point_0} - {self.point_1} - {self.point_2}]"

    def __repr__(self):
        return f"T=[{repr(self.point_0)}-" \
               f"{repr(self.point_1)}-{repr(self.point_2)}]]"

    def __iter__(self):
        point_lst = [self.point_0, self.point_1, self.point_2]
        return iter(point_lst)

    # def __hash__(self):
    #     return hash(self.id)
    #
    # def __eq__(self, other):
    #     if not isinstance(other, Triangle):
    #         raise TypeError("Операнд справа должен иметь тип Triangle")
    #     if hash(self) == hash(other) or self.id is None or other.id is None:
    #         return (self.point_0 == other.point_0) and \
    #                (self.point_1 == other.point_1) and \
    #                (self.point_2 == other.point_2)
    #     return False

    # def get_z_from_xy(self, x, y):
    #     """
    #     Рассчитывает отметку точки (x, y) в плоскости треугольника
    #     :param x: координата x
    #     :param y: координата y
    #     :return: координата z для точки (x, y)
    #     """
    #     a = -((self.point_1.Y - self.point_0.Y) * (self.point_2.Z - self.point_0.Z) -
    #           (self.point_2.Y - self.point_0.Y) * (self.point_1.Z - self.point_0.Z))
    #     b = ((self.point_1.X - self.point_0.X) * (self.point_2.Z - self.point_0.Z) -
    #          (self.point_2.X - self.point_0.X) * (self.point_1.Z - self.point_0.Z))
    #     c = -((self.point_1.X - self.point_0.X) * (self.point_2.Y - self.point_0.Y) -
    #           (self.point_2.X - self.point_0.X) * (self.point_1.Y - self.point_0.Y))
    #     d = -(self.point_0.X * a + self.point_0.Y * b + self.point_0.Z * c)
    #     try:
    #         z = (a * x + b * y + d) / -c
    #     except ZeroDivisionError:
    #         return None
    #     return z
    #
    # def get_area(self):
    #     """
    #     Рассчитывает площадь проекции треугольника на горизонтальной плоскости
    #     """
    #     a = ((self.point_1.X - self.point_0.X)**2 + (self.point_1.Y - self.point_0.Y)**2) ** 0.5
    #     b = ((self.point_2.X - self.point_1.X)**2 + (self.point_2.Y - self.point_1.Y)**2) ** 0.5
    #     c = ((self.point_0.X - self.point_2.X)**2 + (self.point_0.Y - self.point_2.Y)**2) ** 0.5
    #     p = (a + b + c) / 2
    #     geron = (p * (p - a) * (p - b) * (p - c))
    #     s = geron ** 0.5 if geron > 0 else 0
    #     return s
    #
    # def is_point_in_triangle(self, point: Point):
    #     """
    #     Проверяет попадает ли точка внутрь треугольника по критерию суммы площадей
    #     :param point: точка для которой выполняется проверка
    #     :return: True - если точка внутри треугольника и False если нет
    #     """
    #     s_abc = self.get_area()
    #     if s_abc == 0:
    #         return False
    #     s_ab_p = Triangle(self.point_0, self.point_1, point).get_area()
    #     s_bc_p = Triangle(self.point_1, self.point_2, point).get_area()
    #     s_ca_p = Triangle(self.point_2, self.point_0, point).get_area()
    #     delta_s = abs(s_abc - (s_ab_p + s_bc_p + s_ca_p))
    #     if delta_s < 1e-6:
    #         return True
    #     return False
