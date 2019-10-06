import torch
import numpy as np
import threading
import time
import random
import math

import __init__
from abstract_gym.learning.mc_Q_net import Q_net, Q_net2, Q_net3, Q_net10, choose_action
from abstract_gym.environment.occupancy_grid import OccupancyGrid
from abstract_gym.scenario.scene_0 import Scene
from abstract_gym.learning.QT_opt import  BellmanUpdater, EpsilonGreedyPolicyFunction, RingBuffer, RingOfflineData


def labeler_thread_function(q_net, offline_data_ring_buffer, ring_buffer):
    """
    The labeler thread pulls data from offline data set and label it using bellman updater.
    Then the labeled data is saved in the training buffer.
    :param q_net:
    :param offline_data_ring_buffer:
    :param ring_buffer:
    :return:
    """
    print("Starting labeler thread...")
    batch_size = 1000

    while True:
        sars = offline_data_ring_buffer.sample(batch_size=batch_size)
        if len(sars) == 0:
            continue
        bellman_updater = BellmanUpdater(q_net, sars, stage=1)
        sa, qt = bellman_updater.get_labeled_data()
        ring_buffer.insert_labeled_data(sa, qt)


def training_thread_function(q_net, ring_buffer):
    """
    This Thread updates the Q network parameters by minimizing the error of current output Q value and the labeled data.
    :param ring_buffer:
    :param q_net:
    :return:
    """
    print("Starting Training thread...")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4, weight_decay=1e-5)
    while True:
        if len(ring_buffer) > 0:
            saqt = ring_buffer.sample()
            x = saqt[0][0]
            y = saqt[0][1]
            x = x.to(device, dtype=torch.float)
            y = y.view(-1, 1).to(device, dtype=torch.float)
            try:
                q_net.lock.acquire()
                out = q_net.forward(x)
                loss = criterion(y, out).to(device, dtype=torch.float)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            finally:
                q_net.lock.release()
        else:
            time.sleep(0.1)


if __name__ == "__main__":
    threads = list()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = Q_net10().to(device)
    #Q.load_state_dict(torch.load("../models/q_net2_v10_523119_oracle_e1_158.pth"))
    occ = OccupancyGrid(random_obstacle=False)
    scene = Scene(env=occ, visualize=False)
    scene.random_valid_pose()
    record = []
    record_list = []
    file_path = '../data/online_data_list_stage1_1.txt'
    offline_data = RingOfflineData(None, capacity=np.int64(1e7), stage=1)
    buffer = RingBuffer(capacity=np.int64(1e5))

    """
    Starting multi-thread
    """
    labeler_thread = threading.Thread(target=labeler_thread_function, args=(Q, offline_data, buffer))
    training_thread = threading.Thread(target=training_thread_function, args=(Q, buffer))

    threads.append(training_thread)
    threads.append(labeler_thread)

    labeler_thread.start()
    training_thread.start()

    print("Starting main thread...")
    step = 0
    state = np.zeros((4))
    epsilon = 0.0
    succ = 0
    epoches = 0
    max_stage = 100000
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = np.int64(1e8)
    total_steps = np.int64(0)
    last_epoch = 0
    last_succ = 0
    time_out = 0
    #target_c = scene.set_random_target_c()
    for k in range(max_stage):
        #epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * total_steps / EPS_DECAY)
        epsilon = 0.3
        step = 0
        record.clear()
        scene.reset()
        target_c = scene.set_random_target_c()
        #print("Epsilon : ", epsilon)
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
                if np.random.rand() > epsilon:
                    action = scene.inverse_kinematics_controller()
                else:
                    action = scene.sample_action()
                #policy = EpsilonGreedyPolicyFunction(Q, state, epsilon=epsilon)
                #action = policy.choose_action()
            j1, j2, step_reward, done, collision = scene.step(action)
            state = np.array([j1, j2, ik_solution[0], ik_solution[1]])
            record.append(
                [j1, j2, action[0], action[1], step_reward, ik_solution[0], ik_solution[1]]
            )
            if step != 1 and step <= 100:
                offline_data.insert_sars([last_j1, last_j2, ik_solution[0], ik_solution[1], action[0], action[1], step_reward, j1, j2])
            if done:
                succ += 1
                step = 0
                epoches += 1
                record_list.append(record.copy())
                offline_data.insert_record(record.copy())
                record.clear()
                scene.reset()
                target_c = scene.set_random_target_c()
            if collision:
                step = 0
                epoches += 1
                offline_data.insert_sars([last_j1, last_j2, ik_solution[0], ik_solution[1], action[0], action[1], step_reward, j1, j2])
                record_list.append(record.copy())
                offline_data.insert_record(record.copy())
                record.clear()
                scene.reset()
                target_c = scene.set_random_target_c()
            if step > 100:
                step = 0
                time_out += 1
                for _ in range(2):
                    offline_data.insert_sars([last_j1, last_j2, ik_solution[0], ik_solution[1], action[0], action[1], -9900, j1, j2])
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
        exp_num = 2
        model_file_name = "q_net2_v10_" + str(len(record_list)) + "_oracle_e"+str(exp_num)+'_' + str(k) + ".pth"
        torch.save(Q.state_dict(), "../models/" + model_file_name)
        print("Model file: " + model_file_name + " saved.")
        offline_data.write_file('../data/online_data_list_stage1_oracle3.txt')

    for index, thread in enumerate(threads):
        thread.join()

    print("End.")

