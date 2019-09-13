class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y


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
