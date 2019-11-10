import torch
import numpy as np
from torch.utils import data
import threading
import time
import random

import __init__
from abstract_gym.learning.mc_Q_net import choose_action, GeneralNet
from abstract_gym.utils.mysql_interface import Trial_data, SAR_point, Saqt_point, MySQLInterface, TrialDataSampler, \
    DataPointsSampler, SaqtDataSampler
from abstract_gym.learning.QT_opt import BellmanUpdater2, EpsilonGreedyPolicyFunction, RingBuffer, RingOfflineData
from abstract_gym.utils.db_data_type import Trial_data, SAR_point, Saqt_point


def save_model(net, epoch, iter=0):
    model_file_name = "GN818128_v3_" + str(epoch) + "_" + str(iter) + ".pth"
    torch.save(net.state_dict(), '../models/' + model_file_name)
    print("Model file: " + model_file_name + " saved.")


def labeler_thread_function(net, ring_buffer, batch_size=1000):
    """
    The labeler thread pulls data from offline data set and label it using bellman updater.
    Then the labeled data is saved in the training buffer.
    :param net:
    :param batch_size:
    :return:
    """
    db = MySQLInterface()
    labeler_iter = np.int64(0)
    #index = 1
    while True:
        #sarst = db.visit_sarst(batch_size, index)
        sarst = db.sample_sarst(batch_size)
        sarst = torch.DoubleTensor(sarst)
        bellman_updater = BellmanUpdater2(net, sarst)
        saqt_list = bellman_updater.get_labeled_data()
        db.insert_labeled_saqt_data(saqt_list)
        ring_buffer.insert_labeled_saqt_batch(saqt_list)
        #index += batch_size
        #print("index:", index)
        labeler_iter += 1
        if labeler_iter % 1e3 == 0:
            print("labeler:", labeler_iter)


def training_thread_function(net, batch_size=1000):
    """
    This Thread updates the Q network parameters by minimizing the error of current output Q value and the labeled data.
    :param batch_size:
    :param net:
    :return:
    """
    print("Starting Training thread...")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    db = MySQLInterface()
    sampler = SaqtDataSampler(db)
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 1}
    print("training dataset size:", len(sampler))
    training_generator = data.DataLoader(sampler, **params)
    epoch = 0
    iter = np.int64(0)
    try:
        while True:
            for x, y in training_generator:
                x = torch.cat((x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]), 0)
                x = x.reshape(-1, batch_size)
                x = torch.transpose(x, 1, 0)
                x = x.to(device, dtype=torch.float)
                y = y.view(-1, 1).to(device, dtype=torch.float)
                try:
                    net.lock.acquire()
                    out = net.forward(x)
                    loss = criterion(y, out).to(device, dtype=torch.float)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                finally:
                    net.lock.release()
                    if iter % 10 == 0:
                        print("epoch {}, iter {}, loss {:.4f}".format(epoch, iter, loss.data))
                    if iter % 100 == 0:
                        save_model(net, epoch, iter)
                    iter += 1
            print("epoch {}, loss {:.4f}".format(epoch, loss.data))
            iter = 0
            epoch += 1
            save_model(net, epoch)
    except (KeyboardInterrupt, SystemExit):
        save_model(net, epoch)
    except:
        print("Error getting x ,y :", x.shape, y.shape)

def saqt_list_2_xy_tensor(saqt_list):
    x = []
    y = []
    for saqt in saqt_list:
        sa = [saqt[0].j0s, saqt[0].j1s, saqt[0].v0s, saqt[0].v1s, saqt[0].j0t, saqt[0].j1t, saqt[0].a0, saqt[0].a1]
        qt = [saqt[0].qt]
        x.append(sa)
        y.append(qt)
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    return x, y


def training_thread_function2(net, ring_buffer, batch_size=1000):
    """
    This Thread updates the Q network parameters by minimizing the error of current output Q value and the labeled data.
    :param batch_size:
    :param net:
    :return:
    """
    print("Starting Training thread...")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    epoch = 0
    iter = np.int64(0)
    try:
        while True:
            if len(ring_buffer) > batch_size:
                saqt = ring_buffer.sample(batch_size)
                x, y = saqt_list_2_xy_tensor(saqt)
                x = x.to(device, dtype=torch.float)
                y = y.view(-1, 1).to(device, dtype=torch.float) * 0.01
                try:
                    net.lock.acquire()
                    out = net.forward(x)
                    loss = criterion(y, out).to(device, dtype=torch.float)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                finally:
                    net.lock.release()
                    if iter % 10 == 0:
                        print("epoch {}, iter {}, loss {:.4f}".format(epoch, iter, loss.data))
                    if iter % 1000 == 0:
                        save_model(net, epoch, iter)
                    iter += 1
                if iter > 10000:
                    print("epoch {}, loss {:.4f}".format(epoch, loss.data))
                    iter = 0
                    epoch += 1
                    save_model(net, epoch)
            else:
                time.sleep(0.1)
    except (KeyboardInterrupt, SystemExit):
        save_model(net, epoch)
    except:
        print("Error getting x ,y :", x.shape, y.shape)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    GeneralNet(input_size, output_size, hidden_layers, hidden_neurons)
    """
    net = GeneralNet(8, 1, 8, 128).to(device)
    exsit_model_file = "GN818128_v3_435_7000.pth"
    net.load_state_dict(torch.load("../models/" + exsit_model_file))
    print("Load model file: ", exsit_model_file)
    #x =torch.Tensor([6.06778 , 4.87865 , 0 , 0 , 0.302392 , 0.00491547 , 0.0172434 , 0.0440834]).to(device, dtype=torch.float)
    #out = net.forward(x)
    #print(out)

    threads = list()
    buffer = RingBuffer(capacity=np.int64(1e7))
    """
    Starting multi-thread
    """
    labeler_thread = threading.Thread(target=labeler_thread_function, args=(net, buffer))
    training_thread = threading.Thread(target=training_thread_function2, args=(net, buffer))

    threads.append(labeler_thread)
    threads.append(training_thread)

    labeler_thread.start()
    training_thread.start()

    for index, thread in enumerate(threads):
        thread.join()

    print("End.")
