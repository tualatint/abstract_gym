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
        self.resolution = resolution
        self.list_of_line_points = self.l.divide_into_point_set(resolution)


    def obstacle_repulsive_force(self):
        force_list = []
        for p in self.list_of_line_points:
            r = euclidean_distance_square(self.obstacle_center, p)
            f = self.force_coeff * 1.0 / r * np.array((p.x - self.obstacle_center.x), (p.y - self.obstacle_center.y))
            force_list.append(f)
        force_list = np.array(force_list)
        normalized_total_force = force_list.sum(axis=1) / self.resolution
        return force_list, normalized_total_force