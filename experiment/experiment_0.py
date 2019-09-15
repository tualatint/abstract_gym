import time
import threading
import numpy as np

import __init__
from abstract_gym.robot.two_joint_robot import TwoJointRobot
from abstract_gym.environment.occupancy_grid import OccupancyGrid
from abstract_gym.scenario.scene_0 import Scene

"""
Multi-thread version for monte-carlo data collecting.
"""


def thread_function(thread_index, r_list):
    """
    This function runs an instance of data collecting process. It runs in a loop over total_steps, and record a whole
    trajectory end with either successfully reach the target or collision with obstacles.
    :param thread_index: the thread index number.
    :param r_list: the list that passed as reference to store the data.
    :return:
    """
    print("Starting thread :", thread_index)
    rob = TwoJointRobot()
    occ = OccupancyGrid(size=9, random_obstacle=False, obstacle_probability=0.01)
    s = Scene(robot=rob, env=occ, visualize=False)
    s.random_valid_pose()
    record = []
    record_list = []
    succ = 0.0
    total_steps = np.int64(1e5)
    for i in range(total_steps):
        action = s.sample_action(scale_factor=0.1)
        j1, j2, step_reward, done, collision = s.step(action)
        record.append(
            [float(j1), float(j2), float(action[0]), float(action[1]), step_reward]
        )
        if done:
            succ += 1.0
            print("-----------------------!!!!!!!!!!!!!!  UNBELIEVABLE !!!!!!!!!!!!!!!!-------------------------------")
        if done or collision:
            record_list.append(record.copy())
            record.clear()
            print("Thread " + str(thread_index) + " reset at step ", i)
            s.reset()
    print("Thread " + str(thread_index) + " finished :")
    print("Records :", len(record_list))
    print("Success rate :", succ/len(record_list))
    r_list[thread_index] = record_list


if __name__ == "__main__":
    start = time.time()
    threads = list()
    thread_num = 8
    multi_list = [None] * thread_num
    """
    Starting multi-thread
    """
    for index in range(thread_num):
        x = threading.Thread(target=thread_function, args=(index, multi_list))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        thread.join()
    """
    Save the data.
    """
    with open('data_list_8e5.txt', 'w') as f:
        for l in multi_list:
            for d in l:
                f.write("%s\n" % d)

    end = time.time()
    print("Total time comsuption:", end - start)
