import torch
import __init__
from abstract_gym.learning.mc_Q_net import Q_net, choose_action
from abstract_gym.environment.occupancy_grid import OccupancyGrid
from abstract_gym.scenario.scene_0 import Scene
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q = Q_net().to(device)
Q.load_state_dict(torch.load("../models/q_net_v1_993_2.pth"))
occ = OccupancyGrid(random_obstacle=False)
s = Scene(env=occ, visualize=True)
s.random_valid_pose()
record = []
record_list = []
succ = False
action_pool = 500
state = np.zeros((2))
for i in range(np.int64(1e5)):
    al = []
    for k in range(action_pool):
        a = s.sample_action(scale_factor=0.1)
        al.append(a)
    if i == 0:
        action = al[0]
    else:
        action = choose_action(Q, state, al)
    j1, j2, step_reward, done, collision = s.step(action)
    state = np.array([j1, j2])
    record.append(
        [j1, j2, action[0], action[1], step_reward, done, collision]
    )
    print("j1, j2, action, step_reward, done, collision : ", j1, j2, action[0], action[1], step_reward, done, collision)
    if done:
        succ = True
        print("-----------------------!!!!!!!!!!!!!!  UNBELIEVABLE !!!!!!!!!!!!!!!!-------------------------------")
    if done or collision:
        record_list.append(record.copy())
        record.clear()
        print("reset at step ", i)
        s.reset()