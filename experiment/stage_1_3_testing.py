import torch
import numpy as np
from torch.utils import data
import threading
import time
import random

import __init__
from abstract_gym.learning.mc_Q_net import GeneralNet
from abstract_gym.utils.db_data_type import Trial_data, SAR_point, Saqt_point
from abstract_gym.utils.mysql_interface import MySQLInterface, TrialDataSampler
from abstract_gym.robot.two_joint_robot import TwoJointRobot
from abstract_gym.environment.occupancy_grid import OccupancyGrid
from abstract_gym.utils.collision_checker import CollisionChecker
from abstract_gym.utils.geometry import Point, Line
from abstract_gym.utils.repulsive_force import RepulsiveForce
from abstract_gym.utils.PRM import Vertex, Edge, PRM
from abstract_gym.scenario.scene_3 import Scene

class Planer:
    def __init__(self, scene, connectivity_net, value_net):
        self.scene = scene
        self.connectivity_net = connectivity_net
        self.value_net = value_net
        self.uniform_nodes = []
        self.num_uniform_sample = 2000
        self.epsilon = 1e-4
        self.path = []

    def uniform_sample(self):
        for i in range(self.num_uniform_sample):
            j1, j2 = self.scene.random_virtual_valid_pose()
            v = Vertex(j1, j2)
            self.uniform_nodes.append(v)
        return self.uniform_nodes

    def prepare_connectivity_input(self, start_point=None):
        if start_point is None:
            start_point_to_evaluate = self.start_point
        else:
            start_point_to_evaluate = start_point
        x0 = torch.Tensor([start_point_to_evaluate.j1, start_point_to_evaluate.j2])
        x2 = torch.Tensor([self.goal_point.j1, self.goal_point.j2])
        x0 = x0.repeat(self.num_uniform_sample, 1)
        x2 = x2.repeat(self.num_uniform_sample, 1)
        x1 = torch.zeros((self.num_uniform_sample, 2))
        s_g = torch.Tensor([self.start_point.j1, self.start_point.j2, self.goal_point.j1, self.goal_point.j2])
        for idx, n in enumerate(self.uniform_nodes):
            x1[idx] = torch.Tensor([n.j1, n.j2])
        xs = torch.cat((x0, x1), 1)
        xg = torch.cat((x1, x2), 1)
        xs = torch.cat((xs, s_g.unsqueeze(0)), 0)
        xg = torch.cat((xg, s_g.unsqueeze(0)), 0)
        return xs, xg

    def dummy_epsilon(self):
        return (torch.rand(self.num_uniform_sample, 2) - 0.5)*self.epsilon

    def prepare_value_input(self):
        # x = [j0s, j1s, v0s, v1s, j0t, j1t, a0, a1]
        x0s = torch.Tensor([self.start_point.j1, self.start_point.j2])
        x0g = torch.Tensor([self.goal_point.j1, self.goal_point.j2])
        x0s = x0s.repeat(self.num_uniform_sample, 1)
        x0g = x0g.repeat(self.num_uniform_sample, 1)
        node_points = torch.zeros((self.num_uniform_sample, 2))
        for idx, n in enumerate(self.uniform_nodes):
            node_points[idx] = torch.Tensor([n.j1, n.j2])
        x0 = torch.cat((x0s, self.dummy_epsilon(), node_points, self.dummy_epsilon()), 1)
        x1 = torch.cat((node_points, self.dummy_epsilon(), x0g, self.dummy_epsilon()), 1)
        return x0, x1

    def resample_on_weight(self, value_array):
        pass

    def connectivity_mask(self, connectivity_array):
        z = connectivity_array.clone()
        threshold = 2.0
        z[connectivity_array >= threshold] = 1.
        z[connectivity_array < threshold] = 0.
        #print("connectivity mask : ", z)
        return z

    def evaluate_connectivity(self, start_point=None):
        x0, x1 = self.prepare_connectivity_input(start_point)
        x0 = x0.to(device, dtype=torch.float)
        x1 = x1.to(device, dtype=torch.float)
        out0 = self.connectivity_net.forward(x0)
        out1 = self.connectivity_net.forward(x1)
        #print("connectivity out", out)
        m = torch.nn.Sigmoid()
        y0 = m(out0)
        y1 = m(out1)
        out0 = self.connectivity_mask(out0)
        out1 = self.connectivity_mask(out1)
        out = self.connectivity_mask(out1+out0)
        #print("y", y)
        if out[-1, :] == 1.0:
            return out, True
        return out, False

    def evaluate_v_score(self):
        x0, x1 = self.prepare_value_input()
        x0 = x0.to(device, dtype=torch.float)
        x1 = x1.to(device, dtype=torch.float)
        out0 = self.value_net.forward(x0)
        out1 = self.value_net.forward(x1)
        #print("out0 , out1", out0, out1)
        out = out0 + out1
        #print("out ", out)
        return out


    def generate_path(self):
        self.uniform_sample()
        connectivity_array, direct_connected = self.evaluate_connectivity()
        if direct_connected:
            print("Direct connected path.")
            self.path.append(self.goal_point)
            return self.path
        value_array = self.evaluate_v_score()
        _, sorted_ind = torch.sort(value_array)
        num_positive = torch.sum(connectivity_array)
        print("connected points sum: ", num_positive)
        for i in sorted_ind:
            if connectivity_array[int(i)] == 1:
                self.path.append(self.uniform_nodes[int(i)])
                print("Adding via point", self.uniform_nodes[int(i)].j1, self.uniform_nodes[int(i)].j2)
                self.path.append(self.goal_point)
                return self.path
        print("Via point not found.")

    def plan(self,start_point, goal_point):
        self.start_point = start_point
        self.goal_point = goal_point
        return self.generate_path()

    def reset(self):
        self.uniform_nodes = []
        self.path = []

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    connectivity_net = GeneralNet(4, 1, 3, 128).to(device)
    exsit_model_file = "GN413128_plan_v1_532_1514.pth"
    connectivity_net.load_state_dict(torch.load("./models/" + exsit_model_file))
    print("Load model file to connectivity_net: ", exsit_model_file)

    value_net = GeneralNet(8, 1, 8, 128).to(device)
    exsit_model_file = "GN818128_v3_1618_5000.pth"
    value_net.load_state_dict(torch.load("../models/" + exsit_model_file))
    print("Load model file to value_net: ", exsit_model_file)

    obs_rate = 0.03
    damping_ratio = np.sqrt(2.0)/2.0
    random_obstacle = False
    vis = True
    occ = OccupancyGrid(random_obstacle=random_obstacle, obstacle_probability=obs_rate)
    scene = Scene(visualize=vis, env=occ)
    step = 0
    succ = 0
    trial = 0
    scene.random_valid_pose()
    repulsive_scale_factor = 0.005
    scale_factor = 0.8
    solution = None
    verbose = False
    reset_flag = False
    planner = Planer(scene, connectivity_net, value_net)

    while True:
        target_c = scene.set_random_target_c()
        solution = scene.choose_collision_free_ik_solution(target_c)
        #start_point = Vertex(1.0, -0.4)
        #goal_point = Vertex(5.0, 3.0)
        #scene.set_to_pose(start_point.j1, start_point.j2)
        start_point = Vertex(scene.robot.joint_1, scene.robot.joint_2)
        goal_point = Vertex(solution[0], solution[1])
        path = planner.plan(start_point, goal_point)
        if path is not None:
            scene.follow_path_controller(path)
        planner.reset()
        scene.random_valid_pose()
    # while True:
    #     step += 1
    #     alpha = 0.6
    #     beta = 1.0
    #     while solution is None:
    #         target_c = scene.set_random_target_c()
    #         solution = scene.choose_collision_free_ik_solution(target_c)
    #         start_point = Vertex(scene.robot.joint_1, scene.robot.joint_2)
    #         goal_point = Vertex(solution[0], solution[1])
    #         #st = time.time()
    #         path = planer.plan(start_point, goal_point)
    #         #et = time.time()
    #         #prm.visualize()
    #         #print("time consumption: ", et - st)
    #
    #     scene.follow_path_controller(path)
    #     # solution = scene.choose_collision_free_ik_solution(target_c)
    #     # target_acc, target_distance = scene.joint_target_acceleration(solution[0], solution[1])
    #     # repulsive_acc = scene.repulsive_acceleration() * repulsive_scale_factor * np.exp(-1.0 * 1.5/target_distance)
    #     # current_speed = scene.current_joint_speed_sum()
    #     # if step > 100 and (step // 100) % 2 == 0 and (step // 50) % 2 == 0:
    #     #     beta = -1.0
    #     #     alpha = 0.1
    #     # random_acc = (np.random.rand(2)-0.5) * 0.3 * (step / 20) * np.exp(-10.0 * current_speed) * np.exp(-1.0 * 0.03 / np.linalg.norm(target_acc))
    #     # total_acc = alpha * random_acc + beta * target_acc #+ repulsive_acc #+ random_acc
    #     # if step % 50 == 0:
    #     #     print("step {}, solution {:.4f} {:.4f}, joint {:.4f} {:.4}, acc :{:.4f} {:.4f}, beta {}".format(step, solution[0], solution[1], scene.robot.joint_1, scene.robot.joint_2, target_acc[0], target_acc[1], beta))
    #     # scene.acceleration_step_with_wall(total_acc, scale_factor, damping_ratio)
    #
    #     if vis:
    #         scene.render()
    #     if scene.collision_check():
    #         if verbose:
    #             print("collision reset.")
    #             # print("repulsive_acc {:.4f}".format(np.linalg.norm(repulsive_acc)))
    #             # print("target_acc {:.4f}".format(np.linalg.norm(target_acc)))
    #             # print("random_acc {:.4f}".format(np.linalg.norm(random_acc)))
    #         reset_flag = True
    #     if scene.check_target_reached():
    #         succ += 1
    #         if verbose:
    #             print("succ reset.")
    #         reset_flag = True
    #     if step > 1000:
    #         if verbose:
    #             print("over step reset.")
    #             # print("repulsive_acc {:.4f}".format(np.linalg.norm(repulsive_acc)))
    #             # print("target_acc {:.4f}".format(np.linalg.norm(target_acc)))
    #             # print("random_acc {:.4f}".format(np.linalg.norm(random_acc)))
    #         reset_flag = True
    #     if reset_flag:
    #         trial += 1
    #         occ = OccupancyGrid(random_obstacle=random_obstacle, obstacle_probability=obs_rate)
    #         scene.updateOcc(occ, vis=vis)
    #         scene.random_valid_pose()
    #         solution = None
    #         step = 0
    #         reset_flag = False
    #     if trial % 100 == 0 and step == 0:
    #             print("succ rate: {:.4f} in {} trials.".format(succ / trial, trial))
