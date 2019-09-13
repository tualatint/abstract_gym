import numpy as np

import __init__
from abstract_gym.utils.geometry import Point, Square


class OccupancyGrid:
    def __init__(self,
                 size=9,
                 random_obstacle=True,
                 obstacle_probability=0.1,
                 environment_size=1.6
                 ):
        """
        This class represents a occupancy grid. Where:
        occ is the abstract matrix form.
        obstacle_list is the list of obstacles represented in square.
        occ_coordinate is the list of obstacles's bottom left point.

        :param size: the whole environment is divided into a size * size occupancy grid.
        :param random_obstacle: Bool, whether random obstacles are generated.
        :param obstacle_probability: if randomly generates obstacles, this value specify the probability of one grid been occupied.
        :param environment_size: side length of the whole environment in meter.
        """
        self.occ = np.zeros((size, size))
        self.obstacle_list = []
        self.occ_coordinate = []
        self.obstacle_side_length = environment_size/(size-1)
        self.environment_size = environment_size
        self.size = size
        if random_obstacle:
            """
            Randomly generate obstacles.
            """
            for i in range(size):
                for j in range(size):
                    if np.random.rand() < obstacle_probability:
                        self.occ[i][j] = 1  # y x
                        self.occ_coordinate.append([j, i])
        else:
            """
            Manually add obstacles. 
            Example:
            """
            self.occ[5][6] = 1
            self.occ[5][7] = 1
            self.occ[2][3] = 1
            self.occ_coordinate.append([6, 5])
            self.occ_coordinate.append([7, 5])
            self.occ_coordinate.append([3, 2])

        self.transform_frame()

    def transform_frame(self):
        """
        Transform the occupancy grid coordinate from the matrix row col index to the robot_0 frame.
        scale, shift center,
        """
        self.occ_coordinate = np.array(self.occ_coordinate) * self.environment_size / (
                    self.size - 1) - self.environment_size / 2.0
        """ 
        flip
        """
        self.occ_coordinate *= [1, -1]

        for oc in self.occ_coordinate:
            s = Square(Point(oc[0], oc[1]), Point(oc[0] + self.obstacle_side_length, oc[1] + self.obstacle_side_length))
            self.obstacle_list.append(s)

    def get_occupancy_grid(self):
        return self.occ, self.occ_coordinate, self.obstacle_list, self.obstacle_side_length

    def load_from_matrix(self, matrix, environment_size=1.6):
        """
        Load the occupancy grid from a numpy matrix.
        :param matrix: numpy 2d array
        :param environment_size: side length of the whole environment in meter.
        :return:
        """
        if matrix.shape[0] != matrix.shape[1]:
            print("The matrix is not square.")
            return

        self.size = matrix.shape[0]
        self.occ = np.copy(matrix)
        self.obstacle_list.clear()
        self.occ_coordinate = []
        self.obstacle_side_length = environment_size/(self.size-1)
        self.environment_size = environment_size
        for i in range(self.size):
            for j in range(self.size):
                if self.occ[i][j] != 0:  # y x
                    self.occ_coordinate.append([j, i])
        self.transform_frame()




if __name__ == "__main__":
    mat = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 1]])
    print(np.where(mat!=0))
    oc = OccupancyGrid(201)
    oc.load_from_matrix(mat)
    occ, occ_c, ob_l, _ = oc.get_occupancy_grid()

    print(occ)
    print(len(occ_c))


