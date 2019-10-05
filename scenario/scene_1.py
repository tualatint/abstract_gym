import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from numpy import linalg as LA
import random
import time

import __init__
from abstract_gym.robot.two_joint_robot import TwoJointRobot
from abstract_gym.environment.occupancy_grid import OccupancyGrid
from abstract_gym.utils.collision_checker import CollisionChecker
from abstract_gym.utils.geometry import Point, Line
from abstract_gym.utils.repulsive_force import RepulsiveForce


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
        self.ik_solution = None
        self.choose_j_tar = False
        self.init_step_reward = -1
        self.step_reward = self.init_step_reward
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
            self.patches = []
            for p in patch_list:
                self.patches.append(self.ax.add_patch(p))
            self.robot_body_line_vis, self.ee_vis, self.target_c_vis, self.virtual_robot_body_line_vis = self.ax.plot(
                [0, self.robot.elbow_p.x, self.robot.EE.x],
                [0, self.robot.elbow_p.y, self.robot.EE.y],
                'o-',
                [self.robot.EE.x], [self.robot.EE.y], 'ro',
                [self.target_c.x], [self.target_c.y], 'go',
                [0, self.robot.virtual_elbow_point(0).x, self.robot.virtual_end_effector(0, 0).x],
                [0, self.robot.virtual_elbow_point(0).y, self.robot.virtual_end_effector(0, 0).y],
                '--',
            )

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
        """
        A naive implement of jacobian based controller, suffers from singularities.
        :param goal:
        :param scale_factor:
        :return:
        """
        jmat = self.robot.Jacobian_matrix()
        EE = self.robot.end_effector()
        displacement = scale_factor * np.array([goal.x - EE.x, goal.y - EE.y])
        displacement = displacement.reshape(2, -1)
        action = np.asmatrix(jmat.transpose() * jmat).getI() * jmat.transpose() * displacement
        return action, np.linalg.det(jmat.transpose() * jmat)

    def choose_collision_free_ik_solution(self, target_cartesian):
        """
        Find the collsion free ik solution for a given cartesian target, if both ik solutions are collision free, choose
        the one with minimum joint distance.
        :param target_cartesian:
        :return:
        """
        s1, s2 = self.robot.inverse_kinematic(target_cartesian)
        if s1 is None or s2 is None:
            return None
        else:
            collision_free_solution_list = []
            if not self.ik_solution_collision_check(s1):
                collision_free_solution_list.append(s1)
            if not self.ik_solution_collision_check(s2):
                collision_free_solution_list.append(s2)
            if len(collision_free_solution_list) == 0:
                print("Both ik solutions are in collision.")
                return None
            else:
                if len(collision_free_solution_list) == 1:
                    self.ik_solution = collision_free_solution_list[0]
                    return self.ik_solution
                else:
                    s, _, _ = self.choose_ik_with_min_j_distance(s1, s2)
                    self.ik_solution = s
                    return self.ik_solution

    def joint_speed_limit(self, action, limit=0.05):
        if action > limit:
            action = limit
        if action < -limit:
            action = -limit
        return action

    def inverse_kinematics_controller(self):
        """
        Generate a step action for a given ik solution.
        The action is scaled to have a norm of 0.0707
        :return:
        """
        action_norm_scale = 0.0708
        joint_speed_limit = 0.05
        action_j1 = self.ik_solution[0] - self.robot.joint_1
        action_j2 = self.ik_solution[1] - self.robot.joint_2
        if abs(action_j1) > np.pi:
            action_j1 *= -1.0
        if abs(action_j2) > np.pi:
            action_j2 *= -1.0
        action_j1 = self.joint_speed_limit(action_j1, joint_speed_limit)
        action_j2 = self.joint_speed_limit(action_j2, joint_speed_limit)
        action = np.array([action_j1, action_j2])
        if np.linalg.norm(action) > action_norm_scale:
            action = action_norm_scale * action/np.linalg.norm(action)
        return action

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
            self.step_reward = -1e4
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
        self.step_reward = self.init_step_reward


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
        if self.ik_solution is not None:
            self.virtual_robot_body_line_vis.set_ydata([0, self.robot.virtual_elbow_point(self.ik_solution[0]).y, self.robot.virtual_end_effector(self.ik_solution[0],self.ik_solution[1]).y])
            self.virtual_robot_body_line_vis.set_xdata([0, self.robot.virtual_elbow_point(self.ik_solution[0]).x, self.robot.virtual_end_effector(self.ik_solution[0],self.ik_solution[1]).x])
        self.fig.canvas.draw()

    def set_target_c(self, target_c):
        """
        Set the cartesian target of the scene.
        :param target_c: a point
        :return:
        """
        self.target_c = target_c
        if self.vis:
            self.render()

    def set_random_target_c(self):
        """
        Set a randomly generated reachable cartesian target, which is at least collision free for one ik solution.
        :return:
        """
        while True:
            target_c = self.generate_random_target_c()
            s1, s2 = self.robot.inverse_kinematic(target_c)
            if s1 is None or s2 is None:
                continue
            if self.ik_solution_collision_check(s1) and self.ik_solution_collision_check(s2):
                continue
            self.set_target_c(target_c=target_c)
            return target_c

    def ik_solution_collision_check(self, solution):
        """
        Check whether in the ik solution pose the robot is in collision with the obstacles.
        :return: Bool, if it is in collision
        """
        l1 = Line(Point(0, 0), self.robot.virtual_elbow_point(solution[0]))
        l2 = Line(self.robot.virtual_elbow_point(solution[0]), self.robot.virtual_end_effector(solution[0], solution[1]))
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
        """
        Generate a random reachable cartesian target.
        :return: a cartesian target represented as a point.
        """
        condition = False
        while not condition:
            x = 2 * np.random.rand() * self.robot.total_length() - self.robot.total_length()
            y = 2 * np.random.rand() * self.robot.total_length() - self.robot.total_length()
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
        d1 = 2 * np.pi - d0
        d = np.min([d0, d1])
        return d

    def choose_ik_with_min_j_distance(self, s1, s2):
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

    def total_external_repulsive_force_on_link(self, l):
        pf_list = []
        tf_list = []
        for ob in self.obstacle_list:
            rf = RepulsiveForce(l, ob, resolution=5)
            fl, total_f, pf, tf = rf.obstacle_repulsive_force()
            pf_list.append(pf)
        pf_list = np.array(pf_list)
        tf_list = np.array(tf_list)
        total_pf = pf_list.sum(axis=0)
        total_tf = tf_list.sum(axis=0)
        return total_pf, total_tf

    def repulsive_acceleration(self):
        l1 = Line(Point(0, 0), self.robot.elbow_point())
        l2 = Line(self.robot.elbow_point(), self.robot.end_effector())
        pf1, tf1 = self.total_external_repulsive_force_on_link(l1)
        pf2, tf2 = self.total_external_repulsive_force_on_link(l2)
        """
        torque_2 is the external torque that acts on link2
        """
        direct_1 = np.sign(np.dot(pf1, l1.normalized_perpendicular_vec()))
        direct_2 = np.sign(np.dot(pf2, l2.normalized_perpendicular_vec()))
        if isinstance(pf1, float):
            direct_1 = 0
        if isinstance(pf2, float):
            direct_2 = 0
        torque_2 = - direct_2 * (np.linalg.norm(pf2) * (0.5 * l2.length()) - direct_1 * np.linalg.norm(pf1) * 0.1)
        torque_1 = - direct_1 * (np.linalg.norm(pf1) * 0.5 * l1.length()) #+ np.dot(tf2, l1.normalized_perpendicular_vec()) * l1.length())
        acc = np.array([torque_1, torque_2])
        return acc

    def updateOcc(self, occ):
        for p in self.patches:
            p.remove()
        self.occ_matrix, self.occ_coord, self.obstacle_list, self.obstacle_side_length = occ.get_occupancy_grid()
        patch_list = self.occ_to_patch()
        self.patches.clear()
        for p in patch_list:
            self.patches.append(self.ax.add_patch(p))
        self.fig.canvas.draw()

if __name__ == "__main__":
    occ = OccupancyGrid(random_obstacle=True, obstacle_probability=0.03)
    scene = Scene(visualize=True, env=occ)
    step = 0
    scene.random_valid_pose()
    scale_factor = 0.05
    while True:
        step += 1
        acc = scene.repulsive_acceleration() * scale_factor
        print("acc", acc)
        scene.robot.acceleration_control_step(acc, k=0.15)
        scene.render()
        if step > 100:
            print("over step reset.")
            occ = OccupancyGrid(random_obstacle=True, obstacle_probability=0.03)
            scene.updateOcc(occ)
            scene.random_valid_pose()
            step = 0

    # while True:
    #     solution = None
    #     while solution is None:
    #         target_c = scene.set_random_target_c()
    #         #target_c = scene.generate_random_target_c()
    #         #scene.set_target_c(target_c)
    #         s1, s2 = scene.robot.inverse_kinematic(scene.target_c)
    #
    #         print("s1, s2: ", s1, s2)
    #         solution, d1, d2 = scene.choose_ik_with_min_j_distance(s1, s2)
    #         print("d1 ,d2 :", d1, d2)
    #     scene.move_to_joint_pose(solution[0], solution[1])

    # target_c = scene.set_random_target_c()
    # while True:
    #     step += 1
    #     if step == 1:
    #         scene.choose_collision_free_ik_solution(target_c)
    #     action = scene.inverse_kinematics_controller()
    #     if LA.norm(action) > 1:
    #         print("action :", LA.norm(action))
    #         #print("det :", det)
    #         scene.render()
    #         time.sleep(2)
    #     scene.robot.move_delta(action[0], action[1])
    #     if scene.collision_check():
    #         print("collision reset.")
    #         scene.random_valid_pose()
    #         target_c = scene.set_random_target_c()
    #         step = 0
    #     if scene.check_target_reached():
    #         print("succ reset.")
    #         scene.random_valid_pose()
    #         target_c = scene.set_random_target_c()
    #         step = 0
    #     if step > 100:
    #         print("over step reset.")
    #         scene.random_valid_pose()
    #         target_c = scene.set_random_target_c()
    #         step = 0
    #     scene.render()
