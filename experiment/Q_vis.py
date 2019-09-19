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
    print("state", state)
    x = torch.repeat_interleave(state, sample_size)
    x = x.reshape(batch_size, -1, sample_size)
    x = torch.transpose(x, 2, 1)
    state_list = x.reshape(-1, 2).to(dtype=torch.float)
    print(state_list)
    input_x = torch.cat((state_list, action_list), 1)
    input_x = input_x.to(device, dtype=torch.float)
    with torch.no_grad():
        q_list = q_net.forward(input_x)
    q_list = q_list.reshape(-1, sample_size)
    max_q, _ = torch.max(q_list, 1)
    return max_q

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = Q_net2().to(device)
    Q.load_state_dict(torch.load("../models/q_net2_v2_62029_53.pth"))
    occ = OccupancyGrid(random_obstacle=False)
    scene = Scene(env=occ, visualize=False)
    q_list = []
    x_list = []
    y_list = []
    j1_list = []
    j2_list = []
    batch_size = 2
    for i in range(np.int64(1)):
        xb_list = []
        yb_list = []
        j1b_list = []
        j2b_list = []
        qb_list = []
        print("i ", i)
        for k in range(batch_size):
            scene.random_valid_pose()
            j1 = scene.robot.joint_1
            j2 = scene.robot.joint_2
            EE = scene.robot.end_effector()
            xb_list.append(EE.x)
            yb_list.append(EE.y)
            j1b_list.append(j1)
            j2b_list.append(j2)
        state = torch.Tensor([np.squeeze(j1b_list), np.squeeze(j2b_list)])
        q_value = sample_max_q_value(Q, state, batch_size=batch_size)
        q_value = q_value.to(torch.device("cpu")).numpy()
        q_list += q_value.tolist()
        j1_list += j1b_list
        j2_list += j2b_list
        x_list += xb_list
        y_list += yb_list

    area = 10
    print(q_list)
    print("q_list min max :", np.array(q_list).min(), np.array(q_list).max())
    offset = np.array(q_list) - np.abs(np.array(q_list).min())
    fracs = offset.astype(float) / offset.max()
    plt.scatter(j1_list, j2_list, s=area, c=fracs)
    plt.show()
