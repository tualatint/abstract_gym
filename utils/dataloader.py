from torch.utils import data
import numpy as np


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
            end_reward = record[length - 1][4]
            seed = np.random.randint(0, length)
            if seed == 0:
                last_state = np.zeros((2))
            else:
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

        :param end_reward: the end reward, either -1e3 (collision) or 1e4 (success)
        :return:
        """
        return end_reward * pow(self.gamma, l)


def read_file_into_list(path='../data/data_list.txt'):
    """
    Read the data file, collected by the experiment, parse it into a list.
    It might be slow when the file is over 1 GB.
    :param path: data file path.
    :return: data list.
    """
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        record_list = []
        for line in lines:
            one_epoch = line.strip().split('], [')
            epoch = []
            for one_data_point in one_epoch:
                one_data_point = one_data_point.strip("[]")
                elements = one_data_point.strip().split(',')
                ele = []
                for index, e in enumerate(elements):
                    e = e.strip(" ()")
                    e = float(e)
                    ele.append(e)
                epoch.append(ele)
            record_list.append(epoch)
        return record_list


if __name__ == "__main__":
    r_list = read_file_into_list()
    s = Sampler(r_list)
    for i in range(30):
        print(s[2])
