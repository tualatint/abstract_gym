import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

import __init__
from abstract_gym.robot.two_joint_robot import TwoJointRobot
from abstract_gym.environment.occupancy_grid import OccupancyGrid
from abstract_gym.utils.collision_checker import CollisionChecker
from abstract_gym.utils.geometry import Point, Line


class Scene:
    def __init__(self,
                 robot=TwoJointRobot(),
                 env=OccupancyGrid(),
                 target_c=Point(-0.2, -0.3),
                 visualize=False):
        """
        Initialize the scene with a robot and environment
        :param robot: the robot instance
        :param env: the obstacles represented as occupancy grid
        :param target_c: cartesian target pose
        :param visualize: Bool, whether the scene is rendered
        """
        self.robot = robot
        self.occ_matrix, self.occ_coord, self.obstacle_list, self.obstacle_side_length = env.get_occupancy_grid()
        self.vis = visualize
        self.target_c = target_c
        self.target_j = np.array([1.1, -0.2])
        self.choose_j_tar = False
        self.step_reward = 0
        """
        collision_status: finishes one episode by colliding with obstacles.
        """
        self.collision_status = False
        """
        done: successfully finishes one episode by reaching the target.
        """
        self.done = False
        if self.vis:
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            robot_length = self.robot.total_length()
            env_vis_scale = 1.15
            self.ax.set_xlim(-robot_length * env_vis_scale, robot_length * env_vis_scale)
            self.ax.set_ylim(-robot_length * env_vis_scale, robot_length * env_vis_scale)
            self.ax.grid()
            patch_list = self.occ_to_patch()
            for p in patch_list:
                self.ax.add_patch(p)
            self.robot_body_line_vis, self.ee_vis, self.target_c_vis = self.ax.plot(
                [0, self.robot.elbow_p.x, self.robot.EE.x],
                [0, self.robot.elbow_p.y, self.robot.EE.y],
                'o-',
                [self.robot.EE.x], [self.robot.EE.y], 'ro',
                [self.target_c.x], [self.target_c.y], 'go')

    def collision_check(self):
        """
        Check whether the robot is in collision with the obstacles.
        :return: Bool, if it is in collision
        """
        l1 = Line(Point(0, 0), self.robot.elbow_point())
        l2 = Line(self.robot.elbow_point(), self.robot.end_effector())
        for index, ob in enumerate(self.obstacle_list):
            l1c = CollisionChecker(l1, ob)
            c_1 = l1c.collision_check()
            if c_1:
                return True
            l2c = CollisionChecker(l2, ob)
            c_2 = l2c.collision_check()
            if c_2:
                return True
        return False

    def sample_action(self, scale_factor=0.1):
        """
        Randomly generate a small joint movement
        :param scale_factor: scale the movement down.
        :return: joint movement in radian.
        """
        d1 = (np.random.rand() - 0.5) * scale_factor
        d2 = (np.random.rand() - 0.5) * scale_factor
        return np.array([d1, d2])

    def zero_action(self):
        return np.array([0.0, 0.0])

    def step(self, action):
        """
        Move the robot according to the action and check the result.
        :param action: the joint movement
        :return: the state information of the robot
        """
        self.robot.move_delta(action[0], action[1])
        if self.collision_check():
            self.step_reward = -1e3
            self.collision_status = True
        if self.check_target_reached():
            self.step_reward = 1e4
            self.done = True
        if self.vis:
            self.render()
        return self.robot.joint_1, self.robot.joint_2, self.step_reward, self.done, self.collision_status

    def reset(self):
        """
        Reset the scene, re-initialize the robot to a random valid pose and clear the flags
        :return:
        """
        self.random_valid_pose()
        self.collision_status = False
        self.done = False
        self.step_reward = 0

    def check_target_reached(self):
        """
        Check whether the specified target pose is reached or not, with in the error margin of epsilon.
        If choose_j_tar is True, we check the joint pose in radian.
        Otherwise we check the end effector cartesian pose in meter.
        :return: Bool
        """
        epsilon = 2e-3
        if self.choose_j_tar:
            if abs(self.robot.joint_1 - self.target_j[0]) < epsilon and abs(self.robot.joint_2 - self.target_j[1]) < epsilon:
                return True
            else:
                return False
        else:
            if abs(self.target_c.x - self.robot.end_effector().x) < epsilon and abs(
                    self.target_c.y - self.robot.end_effector().y) < epsilon:
                return True
            else:
                return False

    def render(self):
        """
        Render the scene.
        :return:
        """
        self.robot.EE = self.robot.end_effector()
        self.robot.elbow_p = self.robot.elbow_point()
        self.robot_body_line_vis.set_ydata([0, self.robot.elbow_p.y, self.robot.EE.y])
        self.robot_body_line_vis.set_xdata([0, self.robot.elbow_p.x, self.robot.EE.x])
        self.ee_vis.set_ydata([self.robot.EE.y])
        self.ee_vis.set_xdata([self.robot.EE.x])
        self.fig.canvas.draw()

    def occ_to_patch(self):
        """
        Transform the occupancy grid into patches for visualization purposes.
        :return:
        """
        patch_list = []
        for oc in self.occ_coord:
            vertices = [
                (oc[0], oc[1]),  # left, bottom
                (oc[0], oc[1] + self.obstacle_side_length),  # left, top
                (oc[0] + self.obstacle_side_length, oc[1] + self.obstacle_side_length),  # right, top
                (oc[0] + self.obstacle_side_length, oc[1]),  # right, bottom
                (oc[0], oc[1]),  # ignored
            ]
            codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY,
            ]
            path = Path(vertices, codes)
            patch = patches.PathPatch(path, facecolor='orange', lw=2)
            patch_list.append(patch)
        return patch_list

    def random_valid_pose(self):
        """
        Randomly sample a collision free joint pose.
        :return:
        """
        condition = True
        while condition:
            self.robot.joint_1 = np.random.rand() * np.pi * 2.0
            self.robot.joint_2 = np.random.rand() * np.pi * 2.0
            condition = self.collision_check()
