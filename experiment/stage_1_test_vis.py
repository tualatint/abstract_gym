import torch
import numpy as np
import threading
import time
import math

import __init__
from abstract_gym.learning.mc_Q_net import Q_net, Q_net2, Q_net3, Q_net10, choose_action
from abstract_gym.environment.occupancy_grid import OccupancyGrid
from abstract_gym.scenario.scene_0 import Scene
from abstract_gym.learning.QT_opt import BellmanUpdater, EpsilonGreedyPolicyFunction, RingBuffer, RingOfflineData



if __name__ == "__main__":
    threads = list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = Q_net10().to(device)
    #Q.load_state_dict(torch.load("../models/q_net2_v2_60327_52.pth"))
    Q.load_state_dict(torch.load("../models/q_net2_v10_1815778_oracle_e2_806.pth"))
    #Q.load_state_dict(torch.load("../models/q_net2_v3_69947_38.pth"))
    occ = OccupancyGrid(random_obstacle=False)
    scene = Scene(env=occ, visualize=True)
    scene.random_valid_pose()
    record = []
    record_list = []
    file_path = '../data/online_data_list4.txt'
    offline_data = RingOfflineData(None, capacity=np.int64(7e7))
    buffer = RingBuffer(capacity=np.int64(1e5))

    step = 0
    state = np.zeros((4))
    epsilon = 0.0
    succ = 0
    epoches = 0
    max_stage = 1000
    EPS_START = 0.0
    EPS_END = 0.0
    EPS_DECAY = np.int64(1e8)
    total_steps = np.int64(0)
    last_epoch = 0
    last_succ = 0
    time_out = 0
    for k in range(max_stage):
        target_c = scene.set_random_target_c()
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * total_steps / EPS_DECAY)
        print("Epsilon : ", epsilon)
        for i in range(np.int64(1e5)):
            step += 1
            total_steps += 1
            if step == 1:
                action = scene.zero_action()
                ik_solution = scene.choose_collision_free_ik_solution(target_c)
                while ik_solution is None:
                    target_c = scene.set_random_target_c()
                    ik_solution = scene.choose_collision_free_ik_solution(target_c)
            else:
                policy = EpsilonGreedyPolicyFunction(Q, state, epsilon=epsilon)
                action = policy.choose_action(10)
            j1, j2, step_reward, done, collision = scene.step(action)
            state = np.array([j1, j2, ik_solution[0], ik_solution[1]])
            #record.append(
            #    [j1, j2, action[0], action[1], step_reward]
            #)
            if step != 1 and step <= 100:
                #offline_data.insert_sars([last_j1, last_j2, action[0], action[1], step_reward, j1, j2])
                pass
            if done:
                succ += 1
            if done or collision:
                step = 0
                epoches += 1
                #record_list.append(record.copy())
                #offline_data.insert_record(record.copy())
                record.clear()
                scene.reset()
                target_c = scene.set_random_target_c()
            if step > 100:
                step = 0
                time_out += 1
                #offline_data.insert_sars([last_j1, last_j2, action[0], action[1], -1000, j1, j2])
                #print("state_list.append(np.array([{},{}]))".format(j1, j2))
                record.clear()
                scene.reset()
                target_c = scene.set_random_target_c()
            last_j1 = j1
            last_j2 = j2
        if epoches - last_epoch != 0:
            print("Finished main loop with succ rate : {:.4f} in {} epoches." .format(((succ - last_succ)/(epoches - last_epoch)), (epoches - last_epoch)))
        print("Buffer : ", buffer.__len__())
        print("Offline data : ", offline_data.__len__())
        last_epoch = epoches
        last_succ = succ
        print("time out : ", time_out)
        time_out = 0
        #offline_data.write_file('../data/online_data_list_stage_1_oracle.txt')




