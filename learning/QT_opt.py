import torch
import threading
import numpy as np
import random

import __init__
from abstract_gym.utils.dataloader import Sampler, read_file_into_list, read_file_into_sars_list
from abstract_gym.utils.db_data_type import Trial_data, SAR_point, Saqt_point

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BellmanUpdater:
    """
    Update the Q target according to bellman equation:
    Q_t(s,a) = r + max_a' Q_w(s',a')
    """

    def __init__(self, q_net, sars, stage=0):
        self.q_net = q_net
        self.batch_size = sars.shape[0]
        self.sars = torch.Tensor(sars).to(dtype=torch.float)
        self.stage = stage
        if self.stage == 0:
            self.s1a1 = self.sars[:, 0:4]  # s = j1, j2
            self.s2 = self.sars[:, 5:7]
            self.r1 = self.sars[:, 4]
            self.state_vec_length = 2
        else:
            if self.stage == 1:  # s1 t a r s2 = j11, j21, tx, ty, a1, a2, r, j12, j22
                self.s1a1 = self.sars[:, 0:6]  # s = j1, j2, tx, ty
                self.s2 = torch.cat((self.sars[:, 7:9], self.sars[:, 2:4]), 1)
                self.r1 = self.sars[:, 6]
                self.state_vec_length = 4
            else:
                print("Unspecified stage.")

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
        state_list = x.reshape(-1, self.state_vec_length).to(dtype=torch.float)
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


class BellmanUpdater2:
    """
    update the Q value in scene 1 where the action is acceleration
    Q_t(s,a) = r + max_a' Q_w(s',a')
    """

    def __init__(self, q_net, sarst):
        """
        sarst = (j1c, j2c, a1, a2, r, v11, v12, j1n, j2n, v21, v22, j1t, j2t, primary_id)
        """
        self.q_net = q_net
        self.sarst = sarst
        self.batch_size = sarst.shape[0]
        # s a  = j1, j2, v1, v2, jt1, jt2, a1, a2
        self.s1a1 = torch.cat((self.sarst[:, 0:2], self.sarst[:, 5:7], self.sarst[:, 11:13], self.sarst[:, 2:4]), 1)
        self.s2 = self.sarst[:, 7:13]
        self.r1 = self.sarst[:, 4]
        self.id = self.sarst[:, 13]
        self.state_vec_length = 6

    def calculate_qt(self):
        max_q_value = self.sample_max_q_value()
        ending_mask = torch.where(self.r1 != (-1.0 * torch.ones(self.r1.size()).to(dtype=torch.double)), torch.ones(self.r1.size()), torch.zeros(self.r1.size()))
        normal_mask = torch.where(ending_mask != torch.ones(self.r1.size()), torch.ones(self.r1.size()), torch.zeros(self.r1.size()))
        ending_mask = ending_mask.to(device, dtype=torch.float)
        normal_mask = normal_mask.to(device, dtype=torch.float)
        beta = 0.99  # discount value
        qt = (max_q_value * beta + self.r1.to(device, dtype=torch.float)) * normal_mask + ending_mask * self.r1.to(device, dtype=torch.float)
        return qt

    def sample_max_q_value(self, sample_size=500, scale_factor=0.03):
        '''
        action in range (-0.094247, 0.094247)
        :param sample_size:
        :param scale_factor:
        :return:
        '''
        action_list = (np.random.rand(sample_size * self.batch_size, 2) - 0.5) * 2 * np.pi * scale_factor
        action_list = torch.Tensor(action_list)
        x = torch.repeat_interleave(self.s2, sample_size)
        x = x.reshape(self.batch_size, -1, sample_size)
        x = torch.transpose(x, 2, 1)
        state_list = x.reshape(-1, self.state_vec_length).to(dtype=torch.float)
        input_x = torch.cat((state_list, action_list), 1)
        input_x = input_x.to(device, dtype=torch.float)
        try:
            with torch.no_grad():
                q_list = self.q_net.forward(input_x)
        finally:
            pass
        q_list = q_list.reshape(-1, sample_size)
        max_q, _ = torch.max(q_list, 1)
        return max_q

    def get_labeled_data(self):
        sa = self.s1a1
        qt = self.calculate_qt()
        # saqt = (id, j0s, j1s, v0s, v1s, j0t, j1t, a0, a1, qt):
        saqt_list = []
        for idx in range(self.batch_size):
            saqt = Saqt_point(int(self.id[idx].item()), sa[idx, 0].item(), sa[idx, 1].item(), sa[idx, 2].item(),
                              sa[idx, 3].item(),
                              sa[idx, 4].item(), sa[idx, 5].item(), sa[idx, 6].item(), sa[idx, 7].item(),
                              qt=qt[idx].item())
            saqt_list.append(saqt)
        return saqt_list


class RingOfflineData:

    def __init__(self, data_file_path, capacity=np.int64(1e7), stage=0):
        self.capacity = capacity
        if data_file_path is None:
            self.memory = []
            self.position = np.int64(0)
        else:
            self.memory = read_file_into_sars_list(data_file_path, stage)
            self.position = np.int64(len(self.memory))
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
                    f.write("%s\n" % l)
                self.new_data_to_be_appended.clear()
            finally:
                self.lock.release()


class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.lock = threading.Lock()

    def insert_labeled_data(self, sa, qt):
        """Saves a transition."""
        try:
            self.lock.acquire()
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = [sa, qt]
            self.position = (self.position + 1) % self.capacity
        finally:
            self.lock.release()

    def insert_labeled_saqt_batch(self, saqt_batch):
        """Saves a transition."""
        try:
            self.lock.acquire()
            for saqt in saqt_batch:
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                self.memory[self.position] = [saqt]
                self.position = (self.position + 1) % self.capacity
        finally:
            self.lock.release()

    def sample(self, batch_size=1):
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

    def choose_action(self, sample_size=10):
        if np.random.rand() < self.epsilon:
            action = ((np.random.rand(1, 2) - 0.5) * self.scale_factor).squeeze()
        else:
            action = self.sample_best_action(sample_size=sample_size)
        return action

    def sample_best_action(self, sample_size=10):
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


if __name__ == "__main__":
    # pass
    a = torch.ones(5, 2)
    b = 2 * torch.ones(5, 2)
    c = torch.cat((a, b), 1)
    print("b", b.shape)
    print("c", c)
    # a = np.random.rand(np.int64(1e7), 2)
    # a = a.tolist()
    # b = random.sample(a, 1)
    # print(b)
