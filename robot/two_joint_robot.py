import numpy as np

import __init__
from abstract_gym.utils.geometry import Point


class TwoJointRobot:

    def __init__(self,
                 joint_1=np.random.rand() * np.pi * 2.0,
                 joint_2=np.random.rand() * np.pi * 2.0,
                 link_1=0.4,
                 link_2=0.3):
        """
        Initialize the robot.
        :param joint_1: The initial angle of joint_1 in radian.
        :param joint_2: The initial angle of joint_2 in radian.
        :param link_1: The length of link_1 in meters.
        :param link_2: The length of link_2 in meters.
        """
        self.joint_1 = joint_1
        self.joint_2 = joint_2
        self.link_1 = link_1
        self.link_2 = link_2
        self.EE = self.end_effector()
        self.elbow_p = self.elbow_point()

    def total_length(self):
        return self.link_1 + self.link_2

    def end_effector(self):
        """
        Calculate end effector position using forward kinematics.
        :return: EE point
        """
        EE_x = np.cos(self.joint_1) * self.link_1 + np.cos(self.joint_2) * self.link_2
        EE_y = np.sin(self.joint_1) * self.link_1 + np.sin(self.joint_2) * self.link_2
        return Point(EE_x, EE_y)

    def elbow_point(self):
        """
        Calculate the elbow point using forward kinematics
        :return: elbow point
        """
        elbow_x = np.cos(self.joint_1) * self.link_1
        elbow_y = np.sin(self.joint_1) * self.link_1
        return Point(elbow_x, elbow_y)

    def move_to_joint_pose(self, target_j1, target_j2, steps=100):
        """
        Move to target joint pose in multiple steps.
        :param target_j1: target joint 1 pose
        :param target_j2: target joint 2 pose
        :param steps: total steps to perform the motion
        :return:
        """
        alpha = 1.0 / steps
        init_j1 = self.joint_1
        init_j2 = self.joint_2
        for i in range(steps):
            self.joint_1 += alpha * (target_j1 - init_j1)
            self.joint_2 += alpha * (target_j2 - init_j2)

    def move_delta(self, d1, d2):
        """
        Move a small step.
        :param d1: small motion in joint_1
        :param d2: small motion in joint_2
        :return:
        """
        self.joint_1 += d1
        self.joint_2 += d2
