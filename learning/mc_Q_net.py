import torch
from torch.utils import data
import numpy as np
import threading

import __init__
from abstract_gym.utils.dataloader import Sampler, read_file_into_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Q_net(torch.nn.Module):
    """
    A simple 1 hidden layer NN that maps the input of (state, action) pair to a corresponding q value.
    """

    def __init__(self):
        super(Q_net, self).__init__()
        self.fc1 = torch.nn.Linear(4, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.lock = threading.Lock()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

class Q_net2(torch.nn.Module):
    """
    A simple 1 hidden layer NN that maps the input of (state, action) pair to a corresponding q value.
    """

    def __init__(self):
        super(Q_net2, self).__init__()
        self.fc1 = torch.nn.Linear(4, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.lock = threading.Lock()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x
class Q_net3(torch.nn.Module):
    """
    A simple 1 hidden layer NN that maps the input of (state, action) pair to a corresponding q value.
    """

    def __init__(self):
        super(Q_net3, self).__init__()
        self.fc1 = torch.nn.Linear(4, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 1)
        self.lock = threading.Lock()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        x = self.fc4(x)
        return x

def choose_action(q_net, state, action_list):
    """
    Using q_net to evaluate a list of sampled action in a given state, and choose the action with highest q value.
    :param q_net: the trained Q_net.
    :param state: current joint state.
    :param action_list: a list of sampled action.
    :return: the action with highest q value.
    """
    action_list_length = len(action_list)
    action_list = np.array(action_list)
    state_list = np.tile(state, (action_list_length, 1))
    input_x = torch.cat((torch.Tensor(state_list), torch.Tensor(action_list)), 1)
    input_x = torch.Tensor(input_x).to(device, dtype=torch.float)
    q_list = q_net.forward(input_x)
    best_index = torch.argmax(q_list)
    return action_list[best_index]


if __name__ == "__main__":
    """
    Training Q net using randomly collected data.
    """
    file_path = '../data/data_list_10e6.txt'
    record_list = read_file_into_list(file_path)
    sampler = Sampler(record_list)
    params = {"batch_size": 6400, "shuffle": True, "num_workers": 4}
    training_generator = data.DataLoader(sampler, **params)
    """
    Define network.
    """
    criterion = torch.nn.MSELoss()
    learning_rate = 1e-4
    Q = Q_net().to(device)
    optimizer = torch.optim.Adam(Q.parameters(), lr=learning_rate, weight_decay=1e-5)
    max_epoch = np.int64(1e4)
    """
    Training loop.
    """
    try:
        for epoch in range(max_epoch):
            total_loss = 0.0
            iteration = 0
            for x, y in training_generator:
                iteration += 1
                x = x.to(device, dtype=torch.float)
                y = y.view(-1, 1).to(device, dtype=torch.float)
                out = Q.forward(x)
                loss = criterion(y, out)
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Epoch {}: {:.4f}".format(epoch, total_loss / iteration))
    except (KeyboardInterrupt, SystemExit, RuntimeError):
        model_file_name = "q_net_v1_1e7_1.pth"
        torch.save(Q.state_dict(), "../models/" + model_file_name)
        print("Model file: " + model_file_name + " saved.")

    """
    Save model.
    """
    exp_num = 3
    model_file_name = "q_net_v1_" + str(len(record_list)) + "_" + str(exp_num) + ".pth"
    torch.save(Q.state_dict(), "../models/" + model_file_name)
    print("Model file: " + model_file_name + " saved.")
