import __init__
from abstract_gym.robot.two_joint_robot import TwoJointRobot
from abstract_gym.environment.occupancy_grid import OccupancyGrid
from abstract_gym.scenario.scene_0 import Scene
import numpy as np

occ = OccupancyGrid(size=31, random_obstacle=True, obstacle_probability=0.01)
#occ = OccupancyGrid(random_obstacle=False)
s = Scene(env=occ, visualize=True)
s.random_valid_pose()
record = []
record_list = []
succ = False
for i in range(np.int64(1e5)):
    action = s.sample_action(scale_factor=0.1)
    j1, j2, step_reward, done, collision = s.step(action)
    record.append(
        [j1, j2, action[0], action[1], step_reward, done, collision]
    )
    print("j1, j2, action, step_reward, done, collision : ", j1, j2, action[0], action[1], step_reward, done, collision)
    if done:
        succ = True
        print("-----------------------!!!!!!!!!!!!!!  UNBELIEVABLE !!!!!!!!!!!!!!!!-------------------------------")
    if done or collision:
        record_list.append(record.copy())
        record.clear()
        print("reset at step ", i)
        s.reset()
print("records:", len(record_list))
print("succ:", succ)
with open('data_list.txt', 'w') as f:
    for l in record_list:
        for d in l:
            f.write("%s\n" % d)
