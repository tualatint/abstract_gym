import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

import __init__
from abstract_gym.utils.collision_checker import CollisionChecker
from abstract_gym.utils.geometry import Point, Line, Square


class TwoJointRobot:

    def __init__(self,
                 j1=np.random.rand() * np.pi * 2.0,
                 j2=np.random.rand() * np.pi * 2.0,
                 vis=False,
                 target_j=np.array([1.3, -0.3]),
                 target_c=Point(-0.2, -0.3),
                 choose_j_tar=False
                 ):

        self.joint_1 = j1
        self.joint_2 = j2
        self.link_1 = 0.4
        self.link_2 = 0.3
        self.EE = self.end_effector()
        self.elbow_p = self.elbow_point()

    def end_effector(self):
        """
        Calculate end effector position using forward kinematics.
        :return: EE point
        """
        EE_x = np.cos(self.joint_1) * self.link_1 + np.cos(self.joint_2) * self.link_2
        EE_y = np.sin(self.joint_1) * self.link_1 + np.sin(self.joint_2) * self.link_2
        return Point(EE_x, EE_y)

    def elbow_point(self):
        elbow_x = np.cos(self.joint_1) * self.link_1
        elbow_y = np.sin(self.joint_1) * self.link_1
        return Point(elbow_x, elbow_y)