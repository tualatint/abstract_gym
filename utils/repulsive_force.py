import numpy as np

import __init__
from abstract_gym.utils.geometry import Point, Line, Square, euclidean_distance_square


class RepulsiveForce:
    """
    Calculate the repulsive force from the obstacle to the segment of robot body (line).
    """

    def __init__(self, line, square, resolution=5):
        self.l = line
        self.s = square
        self.obstacle_center = self.s.center
        self.force_coeff = 1.0
        self.list_of_line_points = self.l.divide_into_point_set(resolution)



    def obstacle_repulsive_force(self):
        r2_0 = euclidean_distance_square(self.obstacle_center, self.l.p0)
        r2_1 = euclidean_distance_square(self.obstacle_center, self.l.p1)
        f_0 = self.force_coeff * 1.0 / r2_0
        f_1 = self.force_coeff * 1.0 / r2_1
