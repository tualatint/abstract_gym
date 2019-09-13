from torch.utils import data
import numpy as np


class Sampler(data.Dataset):
    def __init__(self, record_list):
        self.rl = record_list
        self.l = len(record_list)
        self.gamma = 0.99

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        try:
            r = self.rl[index]
            length = len(r)
            end_reward = r[length - 1][4]
            seed = np.random.randint(0, length)
            if seed == 0:
                last_state = np.zeros((2))
            else:
                last_state = np.array(r[seed - 1][0:2])
            future_r = self.cal_exp_f_r(length - seed - 1, self.gamma, end_reward)
            x = np.concatenate((last_state, np.array(r[seed][2:4])), axis=0)
            return x, future_r
        except RuntimeError:
            print("r", r[seed])

    def clear(self):
        del self.rl

    def cal_exp_f_r(self, l, gamma, end_reward):
        return end_reward * pow(gamma, l)


def read_file_into_list(path='../data/data_list.txt'):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        record_list = []
        for line in lines:
            ls = line.strip().split('], [')
            epoch = []
            for l in ls:
                l = l.strip("[]")
                elements = l.strip().split(',')
                ele = []
                for index, e in enumerate(elements):
                    e = e.strip(" ()")
                    if index <= 4:
                        e = float(e)
                    ele.append(e)
                epoch.append(ele)
            record_list.append(epoch)
        return record_list


if __name__ == "__main__":
    record_list = read_file_into_list()
    s = Sampler(record_list)
    for i in range(30):
        print(s[2])
