import numpy as np

import __init__
from abstract_gym.utils.geometry import Point, Line, Square


class CollisionChecker:
    """
    Check whether a line segment is inside a square region
    """

    def __init__(self, line, square):
        """
        Initialize the line and square, and calculate the line function in the form of
        ax + by + c = 0
        :param line: The line segment, given by two end points.
        :param square: The obstacle square, given by bottom left and upper right points.
        """
        self.l = line
        self.s = square
        self.a, self.b, self.c = self.l.compute_line_function()

    def compute_corner_line_value(self):
        """
        compute the 4 corner points of the square feed into the line equation
        :return: the signs of these points
        """
        v1 = self.a * self.s.min_x + self.b * self.s.min_y + self.c
        v2 = self.a * self.s.min_x + self.b * self.s.max_y + self.c
        v3 = self.a * self.s.max_x + self.b * self.s.min_y + self.c
        v4 = self.a * self.s.max_x + self.b * self.s.max_y + self.c
        return np.sign(np.array([v1, v2, v3, v4]))

    def collision_check(self):
        """
        If there are both positive and negative values, that means the infinitely long line passes through the square.
        Then we have to check the line segment.
        :return: Bool, whether there is a collision
        """
        v = self.compute_corner_line_value()
        positive_element = np.array(np.where(v > 0))
        negative_element = np.array(np.where(v < 0))
        if positive_element.shape[1] > 0 and negative_element.shape[1] > 0:
            return self.check_sections()
        else:
            return False

    def check_sections(self):
        """
        Check whether the line segment is inside the square
        a = 0: horizontal line
        b = 0: vertical line
        In general case: compute the 4 intersection points of the line and extended square side lines
        Sort these points and compute lambda of the middle two points of the parameterized function:
        lambda = (x - x_0) /(x_1 - x_0) = (y - y_0)/(y_1 - y_0)
        if lambda is inside range (0, 1) that suggest this intersection point is inside the line segment
        :return: Bool, whether there is a collision
        """
        if self.a == 0:
            if self.l.max_x < self.s.min_x or self.l.min_x > self.s.max_x:
                return False
            else:
                return True
        if self.b == 0:
            if self.l.max_y < self.s.min_y or self.l.min_y > self.s.max_y:
                return False
            else:
                return True
        sx_y_min = (-self.c - self.b * self.s.min_y) / self.a
        p1 = Point(sx_y_min, self.s.min_y)
        sx_y_max = (-self.c - self.b * self.s.max_y) / self.a
        p2 = Point(sx_y_max, self.s.max_y)
        sy_x_min = (-self.c - self.a * self.s.min_x) / self.b
        p3 = Point(self.s.min_x, sy_x_min)
        sy_x_max = (-self.c - self.a * self.s.max_x) / self.b
        p4 = Point(self.s.max_x, sy_x_max)
        pl = [p1, p2, p3, p4]
        pl.sort(key=self.take_x)
        lam1 = self.compute_lambda(pl[1].x)
        lam2 = self.compute_lambda(pl[2].x)
        epsilon = 1e-10
        if 1.0 > lam1 > epsilon or 1.0 > lam2 > epsilon:
            return True
        else:
            return False

    def take_x(self, p):
        return p.x

    def compute_lambda(self, x):
        return (x - self.l.p0.x) / (self.l.p1.x - self.l.p0.x)


if __name__ == "__main__":
    c = CollisionChecker(Line(Point(0, 0), Point(1, 2)), Square(Point(0, 0.8), Point(0.9, 1.4)))
    print("collision check:", c.collision_check())
