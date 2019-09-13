import torch
from torch.utils import data
import numpy as np
import __init__
from abstract_gym.utils.dataloader import Sampler , read_file_into_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Q_net(torch.nn.Module):

    def __init__(self):
        super(Q_net, self).__init__()
        self.fc1 = torch.nn.Linear(4, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


def choose_action(q_net, state, action_list):
    l = len(action_list)
    action_list = np.array(action_list)
    state_list = np.tile(state, (l, 1))
    x = torch.cat((torch.Tensor(state_list), torch.Tensor(action_list)), 1)
    x = torch.Tensor(x).to(device, dtype=torch.float)
    q_list = q_net.forward(x)
    best_index = torch.argmax(q_list)
    return action_list[best_index]


if __name__ == "__main__":
    file_path = '../data/data_list_32e5.txt'
    record_list = read_file_into_list(file_path)
    s = Sampler(record_list)
    params = {"batch_size": 640, "shuffle": True, "num_workers": 4}
    training_generator = data.DataLoader(s, **params)
    max_epoch = 10000
    print("device", device)
    criterion = torch.nn.MSELoss()
    lr = 1e-4
    Q = Q_net().to(device)
    optimizer = torch.optim.Adam(Q.parameters(), lr=lr, weight_decay=1e-5)
    try:
        for epoch in range(max_epoch):
            total_loss = 0.0
            iter = 0
            for x, y in training_generator:
                iter += 1
                x = x.to(device, dtype=torch.float)
                y = y.view(-1, 1).to(device, dtype=torch.float)
                out = Q.forward(x)
                loss = criterion(y, out)
                total_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Epoch {}: {:.4f}".format(epoch, total_loss / iter))
    except (KeyboardInterrupt, SystemExit):
        model_file_name = (
                "q_net_v1_32e5_1.pth"
        )
        torch.save(Q.state_dict(), "../models/" + model_file_name)
        print("Model file: " + model_file_name + " saved.")

    exp_num = 2
    model_file_name = "q_net_v1_" + str(len(record_list)) + "_" + str(exp_num) + ".pth"
    torch.save(Q.state_dict(), "../models/" + model_file_name)
    print("Model file: " + model_file_name + " saved.")