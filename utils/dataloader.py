from torch.utils import data
import numpy as np
import time
import threading
import copy


class Sampler(data.Dataset):
    """
    Randomly sample a data point in the record data list.
    Calculate the expected future reward (Q value) of that data point, and feed into a NN for training.
    gamma: discount factor.
    """

    def __init__(self, record_list):
        self.rl = record_list
        self.record_list_length = len(record_list)
        self.gamma = 0.99

    def __len__(self):
        return self.record_list_length

    def __getitem__(self, index):
        """
        Sample one data point from a given epoch (specified by index).
        :param index: randomly generated through torch.utils.data.DataLoader if shuffle is True.
        :return: a data pair of (last_state, action),(q_value)
        """
        try:
            record = self.rl[index]
            length = len(record)
            end_reward = float(record[length - 1][4])
            seed = np.random.randint(1, length)
            last_state = np.array(record[seed - 1][0:2])
            future_reward = self.calculate_expected_future_reward(length - seed - 1, end_reward)
            x = np.concatenate((last_state, np.array(record[seed][2:4])), axis=0)
            return x, future_reward
        except RuntimeError:
            print("record:", record[seed])

    def calculate_expected_future_reward(self, l, end_reward):
        """
        Calculate the expected future reward
        :param l: the steps between sampled step and end.

        :param end_reward: the end reward, either -1e4 (collision) or 1e4 (success)
        :return:
        """
        return end_reward * pow(self.gamma, l)


class OfflineData(data.Dataset):

    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.data_list = read_file_into_list(data_file_path)
        self.data_list_length = len(self.data_list)
        self.new_data_to_be_appended = []
        self.lock = threading.Lock()

    def __len__(self):
        return self.data_list_length

    def insert_record(self, record):
        try:
            self.lock.acquire()
            self.data_list.append(record)
            self.new_data_to_be_appended.append(record)
            self.data_list_length += 1
        finally:
            self.lock.release()

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

    def __getitem__(self, index):
        """
        Get S, A, R, S'
        :param index:
        :return:
        """
        record = self.data_list[index]
        length = len(record)
        seed = np.random.randint(1, length)
        if seed == length - 1:
            next_state = np.zeros((2))
        else:
            next_state = np.array(record[seed + 1][0:2])
        sars = np.concatenate((record[seed][0:5], next_state), axis=0)
        return sars

def form_sars(ele, last_state_j1, last_state_j2, stage=0):
    if stage == 0:
        return last_state_j1, last_state_j2, ele[2], ele[3], ele[4], ele[0], ele[1]
    if stage == 1:
        return last_state_j1, last_state_j2, ele[5], ele[6], ele[2], ele[3], ele[4], ele[0], ele[1]

def read_file_into_sars_list(path='../data/data_list_10e6.txt', stage=0):
    start = time.time()
    total_data_point_num = np.int64(0)
    with open(path, 'r') as f:
        print("Begin to load data from file : " + path)
        lines = f.read().splitlines()
        record_list = []
        sars_list = []
        for line in lines:
            one_epoch = line.strip().split('], [')
            epoch = []
            steps = 0
            for i, one_data_point in enumerate(one_epoch):
                one_data_point = one_data_point.strip("[]")
                elements = one_data_point.strip().split(',')
                ele = []
                sars = []
                for index, e in enumerate(elements):
                    e = e.strip(" ()")
                    try:
                        e = float(e)
                    except:
                        print("e:",e)
                    if index == 0:
                        current_state_j1 = copy.deepcopy(e)
                    if index == 1:
                        current_state_j2 = copy.deepcopy(e)
                    if index == 4 and e == 0.0:
                        e = -1.0
                    ele.append(e)
                if i != 0:
                    sars_element = form_sars(ele, last_state_j1, last_state_j2, stage)
                    sars.append(sars_element)
                last_state_j1 = current_state_j1
                last_state_j2 = current_state_j2
                if i != 0:
                    sars_list.append(sars)
                epoch.append(ele)
                total_data_point_num += 1
                steps += 1
            if steps > 1:
                record_list.append(epoch)
        print("Finished loading data from file : " + path)
        print("Total data points : ", str(total_data_point_num))
        end = time.time()
        sars_list = np.array(sars_list)
        sars_list = np.squeeze(sars_list)
        sars_list = list(sars_list)
        print("Total time comsuption:", end - start)
        return sars_list



def read_file_into_list(path='../data/data_list_10e6.txt'):
    """
    Read the data file, collected by the experiment, parse it into a list.
    It might be slow when the file is over 1 GB.
    :param path: data file path.
    :return: data list.
    """
    start = time.time()
    total_data_point_num = np.int64(0)
    with open(path, 'r') as f:
        print("Begin to load data from file : " + path)
        lines = f.read().splitlines()
        record_list = []
        for line in lines:
            one_epoch = line.strip().split('], [')
            epoch = []
            steps = 0
            for one_data_point in one_epoch:
                one_data_point = one_data_point.strip("[]")
                elements = one_data_point.strip().split(',')
                ele = []
                for index, e in enumerate(elements):
                    e = e.strip(" ()")
                    e = float(e)
                    ele.append(e)
                epoch.append(ele)
                total_data_point_num += 1
                steps += 1
            if steps > 1:
                record_list.append(epoch)
        print("Finished loading data from file : " + path)
        print("Total data points : ", str(total_data_point_num))
        end = time.time()
        print("Total time comsuption:", end - start)
        return record_list


if __name__ == "__main__":
    pass

