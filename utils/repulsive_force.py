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
        """
        Calculate the force act upon the point set, the total force, decomposed tangential and perpendicular force
        The tangential force points from p0 to p1
        :return:
        """
        force_list = []
        for p in self.list_of_line_points:
            r = euclidean_distance_square(self.obstacle_center, p)
            f = self.force_coeff * 1.0 / r * np.array([(p.x - self.obstacle_center.x), (p.y - self.obstacle_center.y)])
            force_list.append(f)
        force_list = np.array(force_list)
        normalized_total_force = force_list.sum(axis=0) / self.resolution
        tangential_force = np.dot(normalized_total_force, self.l.normalized_line_vec())*self.l.normalized_line_vec()
        perpendicular_force = np.sqrt(pow(np.linalg.norm(normalized_total_force), 2) - pow(np.linalg.norm(tangential_force), 2)) * self.l.normalized_perpendicular_vec()
        if np.dot(self.l.normalized_perpendicular_vec(), normalized_total_force) < 0:
            perpendicular_force *= -1
        return force_list, normalized_total_force, perpendicular_force, tangential_force


if __name__=="__main__":
    line = Line(Point(0.0, 0.0), Point(0.5, 0.1))
    square = Square(Point(0.1, 0.1), Point(0.4, 0.4))
    rf = RepulsiveForce(line, square, resolution=1000)
    fl, total_f, pf, tf = rf.obstacle_repulsive_force()
    print("total force :", total_f)
    print("perpendicular force :", pf)
    print("tengential force :", tf)