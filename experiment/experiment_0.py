import time
import threading
import numpy as np

import __init__
from abstract_gym.robot.two_joint_robot import TwoJointRobot
from abstract_gym.environment.occupancy_grid import OccupancyGrid
from abstract_gym.scenario.scene_0 import Scene


def thread_function(num, r_list):
    print("thread :", num)
    rob = TwoJointRobot()
    occ = OccupancyGrid(size=9, random_obstacle=False, obstacle_probability=0.01)
    s = Scene(robot=rob, env=occ, visualize=False)
    s.random_valid_pose()
    record = []
    record_list = []
    succ = False
    for i in range(np.int64(1e6)):
        action = s.sample_action(scale_factor=0.1)
        j1, j2, step_reward, done, collision = s.step(action)
        record.append(
            [j1, j2, action[0], action[1], step_reward, done, collision]
        )
        #print("j1, j2, step_reward, done, collision : ", j1, j2, step_reward, done, collision)
        if done:
            succ = True
            print("-----------------------!!!!!!!!!!!!!!  UNBELIEVABLE !!!!!!!!!!!!!!!!-------------------------------")
        if done or collision:
            record_list.append(record.copy())
            record.clear()
            print("thread " + str(num) + " reset at step ", i)
            s.reset()
    print("records:", len(record_list))
    print("succ:", succ)
    r_list[num] = record_list


if __name__ == "__main__":
    start = time.time()
    threads = list()
    thread_num = 32
    multi_list = [None]*thread_num
    for index in range(thread_num):
        x = threading.Thread(target=thread_function, args=(index, multi_list))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()

    print("Total records:", len(multi_list))
    with open('data_list_32e6.txt', 'w') as f:
        for l in multi_list:
            for d in l:
                f.write("%s\n" % d)
    end = time.time()
    print(end - start)
