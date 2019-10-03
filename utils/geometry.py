import math


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y


def euclidean_distance(p0, p1):
    d = math.sqrt(pow((p0.x - p1.x), 2) + pow((p0.y - p1.y), 2))
    return d


def euclidean_distance_square(p0, p1):
    d = pow((p0.x - p1.x), 2) + pow((p0.y - p1.y), 2)
    return d


class Line:

    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1


    def compute_line_function(self):
        """
        The function of the line should be in this form: ax + by + c = 0
        :return: a, b, c
        """
        if self.p0.x == self.p1.x:  # vertical line
            b = 0
            a = 1
            c = -self.p0.x
            return a, b, c
        if self.p0.y == self.p1.y:  # horizontal line
            a = 0
            b = 1
            c = -self.p0.y
            return a, b, c
        a = 1.0 / (self.p1.x - self.p0.x)
        b = -1.0 / (self.p1.y - self.p0.y)
        c = self.p0.y / (self.p1.y - self.p0.y) - self.p0.x / (self.p1.x - self.p0.x)
        return a, b, c

    def divide_into_point_set(self, resolution):
        """
        Divide the line into point set with specified resolution.
        :param resolution:
        :return:
        """
        if resolution <= 2:
            return [self.p0, self.p1]
        point_set = []
        for i in range(resolution + 1):
            i = i/resolution
            x = i * (self.p1.x - self.p0.x) + self.p0.x
            y = i * (self.p1.y - self.p0.y) + self.p0.y
            p = Point(x, y)
            point_set.append(p)
        return point_set

        


class Square:

    def __init__(self, pbl, pur):
        """
        Define the square using two corner points
        :param pbl: bottom left
        :param pur: upper right
        """
        self.min_x = pbl.x
        self.min_y = pbl.y
        self.max_x = pur.x
        self.max_y = pur.y
        self.center = Point((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)
