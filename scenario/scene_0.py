import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

import __init__
from abstract_gym.robot.two_joint_robot import TwoJointRobot
from abstract_gym.environment.occupancy_grid import OccupancyGrid
from abstract_gym.utils.collision_checker import CollisionChecker



class Scene:
    def __init__(self, robot, env, collision_checker, visualize=False):
        self.robot = robot
        self.env = env
        self.collision_checker = collision_checker
        self.vis = visualize

    def sample_action(self):
        action = np.random.rand()