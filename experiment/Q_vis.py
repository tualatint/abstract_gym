import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

import __init__
from abstract_gym.learning.mc_Q_net import Q_net, Q_net2, Q_net3, choose_action
from abstract_gym.environment.occupancy_grid import OccupancyGrid
from abstract_gym.scenario.scene_0 import Scene


def sample_max_q_value(q_net, state, sample_size=500, scale_factor=0.1, batch_size=1):
    action_list = (np.random.rand(sample_size * batch_size, 2) - 0.5) * scale_factor
    action_list = torch.Tensor(action_list)
    x = torch.repeat_interleave(state, sample_size)
    x = x.reshape(batch_size, -1, sample_size)
    x = torch.transpose(x, 2, 1)
    state_list = x.reshape(-1, 2).to(dtype=torch.float)
    input_x = torch.cat((state_list, action_list), 1)
    input_x = input_x.to(device, dtype=torch.float)
    with torch.no_grad():
        q_list = q_net.forward(input_x)
    q_list = q_list.reshape(-1, sample_size)
    max_q, _ = torch.max(q_list, 1)
    return max_q

def Q_visualizer(q_net, model_file, batch_size=30000, batch_num=100):
    q_net.load_state_dict(torch.load(model_file))
    occ = OccupancyGrid(random_obstacle=False)
    scene = Scene(env=occ, visualize=False)
    q_list = []
    x_list = []
    y_list = []
    j1_list = []
    j2_list = []
    for i in range(np.int64(batch_num)):
        xb_list = []
        yb_list = []
        j1b_list = []
        j2b_list = []
        print("i :", i)
        for k in range(batch_size):
            scene.random_valid_pose()
            j1 = scene.robot.joint_1
            j2 = scene.robot.joint_2
            EE = scene.robot.end_effector()
            xb_list.append(EE.x)
            yb_list.append(EE.y)
            j1b_list.append(j1)
            j2b_list.append(j2)
        state = torch.transpose(torch.Tensor([np.squeeze(j1b_list), np.squeeze(j2b_list)]), 0, 1)
        q_value = sample_max_q_value(q_net, state, batch_size=batch_size)
        q_value = q_value.to(torch.device("cpu")).numpy()
        q_list += q_value.tolist()
        j1_list += j1b_list
        j2_list += j2b_list
        x_list += xb_list
        y_list += yb_list

    area = 0.1
    print("q_list min max :", np.array(q_list).min(), np.array(q_list).max())
    if np.array(q_list).min() < 0:
        offset = np.array(q_list) + np.abs(np.array(q_list).min())
    else:
        offset = np.array(q_list) - np.abs(np.array(q_list).min())
    fracs = offset.astype(float) / offset.max()
    plt.figure()
    plt.subplot(121)
    plt.scatter(x_list, y_list, s=area, c=fracs)
    plt.subplot(122)
    plt.scatter(j1_list, j2_list, s=area, c=fracs)
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = Q_net3().to(device)
    model_file = "../models/q_net3_v1_17688_167.pth"
    q_vis = Q_visualizer(Q, model_file,3000, 10)