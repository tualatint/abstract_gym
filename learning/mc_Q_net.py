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
        self.net = GeneralNet(4, 1, 0, 64)
        self.lock = threading.Lock()

    def forward(self, x):
        x = self.net(x)
        return x

class Q_net2(torch.nn.Module):
    """
    A simple 1 hidden layer NN that maps the input of (state, action) pair to a corresponding q value.
    """

    def __init__(self):
        super(Q_net2, self).__init__()
        self.net = GeneralNet(4, 1, 1, 64)
        self.lock = threading.Lock()

    def forward(self, x):
        x = self.net(x)
        return x
class Q_net3(torch.nn.Module):
    """
    A simple 1 hidden layer NN that maps the input of (state, action) pair to a corresponding q value.
    """

    def __init__(self):
        super(Q_net3, self).__init__()
        self.net = GeneralNet(4, 1, 2, 64)
        self.lock = threading.Lock()

    def forward(self, x):
        x = self.net(x)
        return x

class Q_net10(torch.nn.Module):
    """
    A simple 1 hidden layer NN that maps the input of (state, action) pair to a corresponding q value.
    state = j1, j2, target_cx, target_cy
    """

    def __init__(self):
        super(Q_net10, self).__init__()
        self.net = GeneralNet(6, 1, 2, 64)
        self.lock = threading.Lock()

    def forward(self, x):
        x = self.net(x)
        return x

class Q_netx(torch.nn.Module):
    """
    A simple 1 hidden layer NN that maps the input of (state, action) pair to a corresponding q value.
    state = j1, j2, target_cx, target_cy
    """

    def __init__(self, neuron_size=128):
        super(Q_netx, self).__init__()
        self.net = GeneralNet(6, 1, 5, 128)

    def forward(self, x):
        x = self.net(x)
        return x

class Plan_net(torch.nn.Module):
    """
    A simple 1 hidden layer NN that maps the input of (state, action) pair to a corresponding q value.
    state = j1, j2, target_cx, target_cy
    """

    def __init__(self):
        super(Plan_net, self).__init__()
        self.net = GeneralNet(4, 2, 2, 128)

    def forward(self, x):
        x = self.net(x)
        return x

class GeneralNet(torch.nn.Module):
    """
    A General template for all FC network with identical hidden neuron size.
    """
    def __init__(self, input_size=4, output_size=2, hidden_layers=5, hidden_neurons=128):
        super(GeneralNet, self).__init__()
        self.fc_in = torch.nn.Linear(input_size, hidden_neurons)
        self.fc = torch.nn.ModuleList([torch.nn.Linear(hidden_neurons, hidden_neurons) for i in range(hidden_layers)])
        self.fc_out = torch.nn.Linear(hidden_neurons, output_size)
        self.lock = threading.Lock()

    def forward(self, x):
        x = self.fc_in(x)
        x = torch.relu(x)
        for l in self.fc:
            x = l(x)
            x = torch.relu(x)
        x = self.fc_out(x)
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
