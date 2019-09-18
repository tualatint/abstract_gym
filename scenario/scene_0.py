import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from numpy import linalg as LA
import time

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
            circle1 = plt.Circle((0, 0), self.robot.link_1 - self.robot.link_2, color='r', fill=False)
            circle2 = plt.Circle((0, 0), self.robot.total_length(), color='r', fill=False)
            self.ax.add_artist(circle1)
            self.ax.add_artist(circle2)
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

    def Jacobian_controller(self, goal, scale_factor=0.1):
        jmat = self.robot.Jacobian_matrix()
        EE = self.robot.end_effector()
        displacement = scale_factor * np.array([goal.x - EE.x, goal.y - EE.y])
        displacement = displacement.reshape(2, -1)
        action = np.asmatrix(jmat.transpose() * jmat).getI() * jmat.transpose() * displacement
        return action, np.linalg.det(jmat.transpose() * jmat)

    def inverse_kinematics_controller(self, goal, scale_factor=0.1):
        s1, s2 = self.robot.inverse_kinematic(goal)
        if s1 is None or s2 is None:
            return
        else:
            # if np.random.rand() > 0.5:
            solution = s1
            # else:
            #     solution = s2
        print("inverse kinematic solution: ", solution)
        return self.robot.move_to_joint_pose_step_action(solution[0], solution[1])

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
            if abs(self.robot.joint_1 - self.target_j[0]) < epsilon and abs(
                    self.robot.joint_2 - self.target_j[1]) < epsilon:
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
        self.target_c_vis.set_ydata([self.target_c.y])
        self.target_c_vis.set_xdata([self.target_c.x])
        self.fig.canvas.draw()

    def set_target_c(self, target_c):
        self.target_c = target_c
        self.render()

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

    def move_to_joint_pose(self, target_j1, target_j2, steps=20):
        """
        Move to target joint pose in multiple steps.
        :param target_j1: target joint 1 pose
        :param target_j2: target joint 2 pose
        :param steps: total steps to perform the motion
        :return:
        """
        alpha = 1.0 / steps
        init_j1 = self.robot.joint_1
        init_j2 = self.robot.joint_2
        target_j1 = self.find_nearest_target(init_j1, target_j1)
        target_j2 = self.find_nearest_target(init_j2, target_j2)
        for i in range(steps):
            self.robot.joint_1 += alpha * (target_j1 - init_j1)
            self.robot.joint_2 += alpha * (target_j2 - init_j2)
            self.render()
            if scene.check_target_reached():
                print("succ reset.")
                self.random_valid_pose()
            if scene.collision_check():
                print("collision reset.")
                scene.random_valid_pose()
        self.robot.joint_range_check()

    def generate_random_target_c(self):
        condition = False
        while not condition:
            x = np.random.rand() * self.robot.total_length() - 0.5
            y = np.random.rand() * self.robot.total_length() - 0.5
            condition = self.robot.cart_target_valid_check(Point(x, y))
        return Point(x, y)

    def find_nearest_target(self, v1, v2):
        d0 = abs(v1 - v2)
        d1 = abs(v1 - v2 + 2 * np.pi)
        d2 = abs(v1 - v2 - 2 * np.pi)
        index = np.argmin([d0, d1, d2])
        v3 = [v2, v2 - 2 * np.pi, v2 + 2 * np.pi]
        return v3[index]

    def find_min_distance(self, v1, v2):
        d0 = abs(v1 - v2)
        d1 = abs(v1 - v2 + 2 * np.pi)
        d2 = abs(v1 - v2 - 2 * np.pi)
        d = np.min([d0, d1, d2])
        return d

    def choose_inv_ik_with_min_j_distance(self, s1, s2):
        if s1 is None or s2 is None:
            return None, None, None
        d11 = self.find_min_distance(self.robot.joint_1, s1[0])
        d12 = self.find_min_distance(self.robot.joint_1, s2[0])
        d21 = self.find_min_distance(self.robot.joint_2, s1[1])
        d22 = self.find_min_distance(self.robot.joint_2, s2[1])
        d1 = d11 + d21
        d2 = d12 + d22
        if d1 > d2:
            return s2, d1, d2
        else:
            return s1, d1, d2


if __name__ == "__main__":
    occ = OccupancyGrid(random_obstacle=False, obstacle_probability=0.1)
    scene = Scene(visualize=True, env=occ)
    step = 0
    scene.random_valid_pose()

    while True:
        solution = None
        while solution is None:
            target_c = scene.generate_random_target_c()
            scene.set_target_c(target_c)
            s1, s2 = scene.robot.inverse_kinematic(scene.target_c)
            print("s1, s2: ", s1, s2)
            solution, d1, d2 = scene.choose_inv_ik_with_min_j_distance(s1, s2)
            print("d1 ,d2 :", d1, d2)
        scene.move_to_joint_pose(solution[0], solution[1])

    # while True:
    #     step += 1
    #     #action, det = scene.Jacobian_controller(goal=scene.target_c)
    #     action = scene.inverse_kinematics_controller(goal=scene.target_c)
    #     if LA.norm(action) > 1:
    #         print("action :", LA.norm(action))
    #         #print("det :", det)
    #         scene.render()
    #         time.sleep(2)
    #     scene.robot.move_delta(action[0], action[1])
    #     if scene.collision_check():
    #         print("collision reset.")
    #         scene.random_valid_pose()
    #         step = 0
    #     if scene.check_target_reached():
    #         print("succ reset.")
    #         scene.random_valid_pose()
    #         step = 0
    #     if step > 100:
    #         print("over step reset.")
    #         scene.random_valid_pose()
    #         step = 0
    #     scene.render()
