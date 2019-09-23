import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.image import NonUniformImage

import __init__
from abstract_gym.learning.mc_Q_net import Q_net, Q_net2, Q_net3, choose_action
from abstract_gym.environment.occupancy_grid import OccupancyGrid
from abstract_gym.scenario.scene_0 import Scene

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def generate_q_spectrum(q_net, state, resolution=1000, batch_size=1):
    x = np.linspace(-0.05, 0.05, resolution)
    y = np.linspace(-0.05, 0.05, resolution)
    action_list = []
    for x_i in x:
        for y_i in y:
            action_list.append([x_i, y_i])
    sample_size = len(action_list)
    action_list = torch.Tensor(action_list)
    x = torch.repeat_interleave(state, sample_size)
    x = x.reshape(batch_size, -1, sample_size)
    x = torch.transpose(x, 2, 1)
    state_list = x.reshape(-1, 2).to(dtype=torch.float)
    input_x = torch.cat((state_list, action_list), 1)
    input_x = input_x.to(device, dtype=torch.float)
    with torch.no_grad():
        q_list = q_net.forward(input_x)
    return q_list

def Q_state_visualizer(q_net, state, resolution=100):
    state = torch.Tensor(state)
    q_value = generate_q_spectrum(q_net, state, resolution=resolution)
    q_value = q_value.to(torch.device("cpu")).numpy()
    q_list = np.reshape(q_value, (resolution, resolution))
    area = 1
    print("q_list min max :", np.array(q_list).min(), np.array(q_list).max())
    if np.array(q_list).min() < 0:
        offset = np.array(q_list) + np.abs(np.array(q_list).min())
    else:
        offset = np.array(q_list) - np.abs(np.array(q_list).min())
    fracs = offset.astype(float) / offset.max()
    plt.matshow(fracs)
    plt.show()

def Q_visualizer(q_net, batch_size=30000, batch_num=100):
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
    Q = Q_net2().to(device)
    model_file = "../models/q_net2_v3_69947_38.pth"
    Q.load_state_dict(torch.load(model_file))
    vis_q = False
    if vis_q:
        q_vis = Q_visualizer(Q, 3000, 10)
    else:
        state_list = []
        state_list.append(np.array([1.152975300810737, 4.24014876965144]))
        # state_list.append(np.array([3.6531312580323227, 3.6013018175557208]))
        # state_list.append(np.array([2.6598438153666173, 0.39138112928921187]))
        # state_list.append(np.array([4.931533496754234, 2.8517578214751858]))
        # state_list.append(np.array([1.5062342785165335, 0.22521968575919593]))
        # state_list.append(np.array([0.3980329837250166, 0.7738175322604637]))
        # state_list.append(np.array([0.7787336366581453, 0.7219527152554378]))


        for state in state_list:
            q_state_vis = Q_state_visualizer(Q, state=state)