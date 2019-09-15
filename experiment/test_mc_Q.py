import torch
import numpy as np

import __init__
from abstract_gym.learning.mc_Q_net import Q_net, choose_action
from abstract_gym.environment.occupancy_grid import OccupancyGrid
from abstract_gym.scenario.scene_0 import Scene
from abstract_gym.learning.QT_opt import TrainingBuffer, BellmanUpdater, TrainingThread, EpsilonGreedyPolicyFunction



def still_check(a):
    epsilon = 2e-3
    if abs(a[0]) < epsilon and abs(a[1]) < epsilon:
        return True
    else:
        return False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q = Q_net().to(device)
Q.load_state_dict(torch.load("../models/q_net_v1_4866_2.pth"))
occ = OccupancyGrid(random_obstacle=False)
scene = Scene(env=occ, visualize=True)
scene.random_valid_pose()
record = []
record_list = []
succ = False
action_pool = 5000
step = 0
state = np.zeros((2))
for i in range(np.int64(1e5)):
    step += 1
    if i == 0:
        action = scene.zero_action()
    else:
        policy = EpsilonGreedyPolicyFunction(Q, state, epsilon=0.0)
        action = policy.choose_action()
    j1, j2, step_reward, done, collision = scene.step(action)
    state = np.array([j1, j2])
    record.append(
        [j1, j2, action[0], action[1], step_reward, done, collision]
    )
    print("j1, j2, action, step_reward, done, collision : ", j1, j2, action[0], action[1], step_reward, done, collision)
    if done:
        succ = True
        print("-----------------------!!!!!!!!!!!!!!  UNBELIEVABLE !!!!!!!!!!!!!!!!-------------------------------")
    if done or collision or still_check(action):
        step = 0
        record_list.append(record.copy())
        record.clear()
        print("reset at step ", i)
        scene.reset()
