import torch
import numpy as np
from torch.utils import data
import threading
import time
import random

import __init__
from abstract_gym.learning.mc_Q_net import GeneralNet
from abstract_gym.utils.db_data_type import Trial_data, SAR_point, Saqt_point
from abstract_gym.utils.mysql_interface import MySQLInterface, TrialDataSampler


def save_model(net, epoch, iter=0):
    model_file_name = "GN413128_plan_v1_" + str(epoch) + "_" + str(iter) + ".pth"
    torch.save(net.state_dict(), './models/' + model_file_name)
    print("Model file: " + model_file_name + " saved.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = GeneralNet(4,1,3,128).to(device)
    exsit_model_file = "GN413128_plan_v1_2.pth"
    net.load_state_dict(torch.load("./models/" + exsit_model_file))
    print("Load model file: ", exsit_model_file)
    db = MySQLInterface()
    params = {"batch_size": 1000, "shuffle": True, "num_workers": 1}
    training_set = TrialDataSampler(db)
    print("training dataset size:", len(training_set))
    training_generator = data.DataLoader(training_set, **params)
    #validation_generator = data.DataLoader(training_set, **params)
    #criterion = torch.nn.MSELoss()
    criterion2 = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    epoch = np.int64(0)
    while True:
        epoch += 1
        iteration = np.int64(0)
        try:
            for x, y in training_generator:
                iteration += 1
                x, y = (
                    torch.stack(x).to(device, dtype=torch.float),
                    torch.stack(y).to(device, dtype=torch.float),
                )
                x = x.transpose(1, 0)
                y = y.transpose(1, 0)
                out = net.forward(x)
                m = torch.nn.Sigmoid()
                loss2 = criterion2(m(out), y[:, 0])
                #loss = criterion(out[0, 1], y[0, 1])
                #total_loss = loss + 100*loss2
                total_loss = loss2
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                print("epoch {} iter {} loss {:.4f}".format(epoch, iteration, total_loss.data))
            if iteration % 1000 == 0:
                save_model(net, epoch, iteration)
            if epoch % 2 == 0:
                save_model(net, epoch, iteration)
        except (KeyboardInterrupt, SystemExit):
            save_model(net, epoch, iteration)