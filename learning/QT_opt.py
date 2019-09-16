import torch
import threading
import numpy as np
import random

import __init__
from abstract_gym.utils.dataloader import Sampler, read_file_into_list, read_file_into_sars_list


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BellmanUpdater:
    """
    Update the Q target according to bellman equation:
    Q_t(s,a) = r + max_a' Q_w(s',a')
    """
    def __init__(self, q_net, sars):
        self.q_net = q_net
        self.batch_size = sars.shape[0]
        self.sars = torch.Tensor(sars).to(dtype=torch.float)
        self.s1a1 = self.sars[:, 0:4]
        self.s2 = self.sars[:, 5:7]
        self.r1 = self.sars[:, 4]

    def calculate_qt(self):
        max_q_value = self.sample_max_q_value()
        qt = max_q_value + self.r1.to(device, dtype=torch.float)
        return qt

    def sample_max_q_value(self, sample_size=500, scale_factor=0.1):
        action_list = (np.random.rand(sample_size * self.batch_size, 2) - 0.5) * scale_factor
        action_list = torch.Tensor(action_list)
        x = torch.repeat_interleave(self.s2, sample_size)
        x = x.reshape(self.batch_size, -1, sample_size)
        x = torch.transpose(x, 2, 1)
        state_list = x.reshape(-1, 2).to(dtype=torch.float)
        input_x = torch.cat((state_list, action_list), 1)
        input_x = input_x.to(device, dtype=torch.float)
        try:
            self.q_net.lock.acquire()
            with torch.no_grad():
                q_list = self.q_net.forward(input_x)
        finally:
            self.q_net.lock.release()
        q_list = q_list.reshape(-1, sample_size)
        max_q, _ = torch.max(q_list, 1)
        return max_q

    def get_labeled_data(self):
        return self.s1a1, self.calculate_qt()


class RingOfflineData:

    def __init__(self, data_file_path, capacity=np.int64(1e7)):
        self.capacity = capacity
        if data_file_path is None:
            self.memory = []
        else:
            self.memory = read_file_into_sars_list(data_file_path)
        self.position = np.int64(0)
        self.data_file_path = data_file_path
        self.new_data_to_be_appended = []
        self.lock = threading.Lock()

    def insert_record(self, record):
        """
        :param record:
        :param init_data:
        :return:
        """
        try:
            self.lock.acquire()
            self.new_data_to_be_appended.append(record)
        finally:
            self.lock.release()

    def insert_sars(self, sars):
        try:
            self.lock.acquire()
            if len(self.memory) < self.capacity:
                self.memory.append(None)
                self.memory[self.position] = sars
                self.position = (self.position + 1) % self.capacity
        finally:
            self.lock.release()

    def sample(self, batch_size):
        """
        TODO: fix the ring data dependency issue
        Get S, A, R, S'
        :return:
        """
        sars_raw_list = []
        try:
            self.lock.acquire()
            if batch_size > len(self.memory):
                return []
            sars_raw_list = random.sample(self.memory, batch_size)
        finally:
            self.lock.release()
            sars_list = [x for x in sars_raw_list if x is not None]
            sars_list = np.array(sars_list)
        return sars_list

    def __len__(self):
        return len(self.memory)

    def write_file(self, path=None):
        if path is None:
            path = self.data_file_path
        with open(path, 'a+') as f:
            try:
                self.lock.acquire()
                for l in self.new_data_to_be_appended:
                    for d in l:
                        f.write("%s\n" % d)
                self.new_data_to_be_appended.clear()
            finally:
                self.lock.release()


class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.lock = threading.Lock()

    def insert_labeled_data(self,  sa, qt):
        """Saves a transition."""
        try:
            self.lock.acquire()
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = [sa, qt]
            self.position = (self.position + 1) % self.capacity
        finally:
            self.lock.release()

    def sample(self, batch_size):
        try:
            self.lock.acquire()
            element = random.sample(self.memory, batch_size)
        finally:
            self.lock.release()
        return element

    def __len__(self):
        return len(self.memory)


class EpsilonGreedyPolicyFunction:

    def __init__(self, q_net, state, epsilon=0.2, scale_factor=0.1):
        self.epsilon = epsilon
        self.q_net = q_net
        self.state = state
        self.scale_factor = scale_factor

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            action = ((np.random.rand(1, 2) - 0.5) * self.scale_factor).squeeze()
        else:
            action = self.sample_best_action()
        return action

    def sample_best_action(self, sample_size=500):
        action_list = (np.random.rand(sample_size, 2) - 0.5) * self.scale_factor
        state_list = np.tile(self.state, (sample_size, 1))
        input_x = torch.cat((torch.Tensor(state_list), torch.Tensor(action_list)), 1)
        input_x = torch.Tensor(input_x).to(device, dtype=torch.float)
        try:
            self.q_net.lock.acquire()
            with torch.no_grad():
                q_list = self.q_net.forward(input_x)
        finally:
            self.q_net.lock.release()
            best_index = torch.argmax(q_list)
        return action_list[best_index]


if __name__ =="__main__":
    pass