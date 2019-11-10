import torch
import numpy as np
import threading
import time
import random

import __init__
from abstract_gym.learning.mc_Q_net import Q_net, Q_net2, Q_net3, Q_net10, choose_action
from abstract_gym.environment.occupancy_grid import OccupancyGrid
from abstract_gym.scenario.scene_1 import Scene
from abstract_gym.learning.QT_opt import BellmanUpdater, EpsilonGreedyPolicyFunction, RingBuffer, RingOfflineData
from abstract_gym.utils.mysql_interface import trial_data, SAR_point, MySQLInterface

if __name__ == "__main__":
    db = MySQLInterface()
    obs_rate = 0.03
    damping_ratio = np.sqrt(2.0)/2.0
    random_obstacle = False
    vis = False
    occ = OccupancyGrid(random_obstacle=random_obstacle, obstacle_probability=obs_rate)
    scene = Scene(visualize=vis, env=occ)
    step = 0
    succ = 0
    trial = 0
    scene.random_valid_pose()
    j1s, j2s = scene.current_joint_pose()
    repulsive_scale_factor = 0.005
    scale_factor = 0.8
    solution = None
    verbose = False
    reset_flag = False
    succ_flag = False
    while True:
        step += 1
        while solution is None:
            target_c = scene.set_random_target_c()
            solution = scene.choose_collision_free_ik_solution(target_c)
        target_acc, target_distance = scene.joint_target_acceleration(solution[0], solution[1])
        current_speed = scene.current_joint_speed()
        j1, j2, step_reward, done, collision = scene.acceleration_step(target_acc, scale_factor, damping_ratio)
        idx = db.get_current_trial_index() + 1
        """
        SAR_point(idx, step, j1, j2, a1, a2, r)
        """
        db.insert_sar_data_point(SAR_point(idx, step, j1, j2, target_acc[0], target_acc[1], step_reward))
        if vis:
            scene.render()
        if scene.collision_check():
            if verbose:
                print("collision reset.")
                print("target_acc {:.4f}".format(np.linalg.norm(target_acc)))
            reset_flag = True
        if scene.check_target_reached():
            succ += 1
            succ_flag = True
            if verbose:
                print("succ reset.")
            reset_flag = True
        if step > 500:
            if verbose:
                print("over step reset.")
                print("target_acc {:.4f}".format(np.linalg.norm(target_acc)))
            reset_flag = True
        if reset_flag:
            trial += 1
            """
            trial_data(Succ, Duration, J1s, J2s, J1e, J2e)
            """
            db.insert_trial(trial_data(succ_flag, step, j1s, j2s, solution[0], solution[1]))
            scene.reset()
            j1s, j2s = scene.current_joint_pose()
            solution = None
            step = 0
            reset_flag = False
            succ_flag = False
        if trial % 100 == 0 and step == 0:
                print("succ rate: {:.4f} in {} trials.".format(succ / trial, trial))