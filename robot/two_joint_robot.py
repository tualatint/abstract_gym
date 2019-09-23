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

    def virtual_end_effector(self, j1, j2):
        """
        Calculate end effector position using forward kinematics.
        :return: EE point
        """
        EE_x = np.cos(j1) * self.link_1 + np.cos(j2) * self.link_2
        EE_y = np.sin(j1) * self.link_1 + np.sin(j2) * self.link_2
        return Point(EE_x, EE_y)

    def elbow_point(self):
        """
        Calculate the elbow point using forward kinematics
        :return: elbow point
        """
        elbow_x = np.cos(self.joint_1) * self.link_1
        elbow_y = np.sin(self.joint_1) * self.link_1
        return Point(elbow_x, elbow_y)

    def virtual_elbow_point(self, j1):
        """
        Calculate the elbow point of a given joint 1 value using forward kinematics
        :return: elbow point
        """
        elbow_x = np.cos(j1) * self.link_1
        elbow_y = np.sin(j1) * self.link_1
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
            self.joint_range_check()

    def move_to_joint_pose_step_action(self, target_j1, target_j2, steps=100):
        alpha = 1.0 / steps
        init_j1 = self.joint_1
        init_j2 = self.joint_2
        action_j1 = alpha * (target_j1 - init_j1)
        action_j2 = alpha * (target_j2 - init_j2)
        return np.array([action_j1, action_j2])

    def move_delta(self, d1, d2):
        """
        Move a small step.
        :param d1: small motion in joint_1
        :param d2: small motion in joint_2
        :return:
        """
        self.joint_1 += d1
        self.joint_2 += d2
        self.joint_range_check()

    def joint_range_check(self):
        """
        If the joint value is out side the range [0, 2*pi], change it back.
        :return:
        """
        self.joint_1 = self.range_check(self.joint_1)
        self.joint_2 = self.range_check(self.joint_2)

    def range_check(self, j):
        """
        If the joint value is out side the range [0, 2*pi], change it back.
        :return:
        """
        if j > np.pi * 2.0:
            j -= np.pi * 2.0
        if j < 0:
            j += np.pi * 2.0
        return j

    def cart_target_valid_check(self, target_c):
        """
        Check the specified target cartesian pose is reachable or not
        :param target_c: specified cartesian pose of end effector
        :return: Bool
        """
        R = self.total_length()
        r = abs(self.link_1 - self.link_2)
        radius = np.sqrt(pow(target_c.x, 2) + pow(target_c.y, 2))
        if r < radius <= R:
            return True, radius
        else:
            return False, radius

    def inverse_kinematic(self, target_c):
        """
        Compute the joint value for a given end effector cartesian pose using cosine theorem.
        :param target_c: end effector cartesian pose.
        :return: joint value j1 , j2
        """
        valid, radius = self.cart_target_valid_check(target_c)
        if valid:
            if radius == 0:
                print("link_1 equals link 2 and the target is at origin, infinite many solutions.")
                return None, None
            cos_theta = (pow(radius, 2) + pow(self.link_1, 2) - pow(self.link_2, 2)) / (2.0 * self.link_1 * radius)
            theta = np.arccos(cos_theta)
            alpha = np.arctan2(target_c.y, target_c.x)
            j1_1 = alpha - theta
            j1_2 = alpha + theta
            cos_beta = (pow(self.link_1, 2) + pow(self.link_2, 2) - pow(radius, 2)) / (2.0 * self.link_1 * self.link_2)
            j2_1 = np.pi - np.arccos(cos_beta) + j1_1
            j2_2 = j1_2 - (np.pi - np.arccos(cos_beta))
            j1_1 = self.range_check(j1_1)
            j1_2 = self.range_check(j1_2)
            j2_1 = self.range_check(j2_1)
            j2_2 = self.range_check(j2_2)
            s1 = np.array([j1_1, j2_1])
            s2 = np.array([j1_2, j2_2])
            return s1, s2
        else:
            #print("Target out of reach.")
            return None, None

    def Jacobian_matrix(self):
        return np.array([np.squeeze([-self.link_1 * np.sin(self.joint_1), -self.link_2 * np.sin(self.joint_2)]),
                         np.squeeze([self.link_1 * np.cos(self.joint_1), self.link_2 * np.cos(self.joint_2)])])



if __name__ == "__main__":
    tar = Point(0.5, 0.0)
    robot = TwoJointRobot()
    J_mat = robot.Jacobian_matrix()
    print("J mat :", J_mat)
    A = J_mat * J_mat.transpose()
    B = J_mat.transpose() * J_mat
    print("A mat :", A)
    print("B mat :", B)
    print("inverse J :", np.asmatrix(J_mat).getI())
    #s1, s2 = robot.inverse_kinematic(tar)
    #print("s1 :", s1)
    #print("s2 :", s2)
