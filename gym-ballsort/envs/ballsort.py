import gym
import numpy as np
import os
import json


class BallSortEnv(gym.Env):
    def __init__(self):
        # Read level config from file
        self.f = open("levels/007-config.json")
        self.json_file = json.load(self.f)

        # Define according to level config
        self.n_tubes = len(self.json_file["tubes"])
        self.n_colors = len(set([color for sublist in self.json_file["tubes"] for color in sublist]))
        self.n_actions = self.n_tubes**2

        self.state = np.zeros((self.n_tubes, 4, self.n_colors))

        ## Define action and state spaces
        # Can make any move between a pair of tubes (although several might be ilegal)
        self.action_space = gym.spaces.Box(shape=(self.n_tubes,self.n_tubes))

        # Get a 3D matrix of shape (shape=(n_tubes, tube_length, n_colors))
        self.observation_space = gym.spaces.Box(shape=(self.n_tubes, 4, self.n_colors))
        

    def step(self, action):
        state = 1
        
        if action == 2:
            reward = 1
        else:
            reward = -1
            
        done = True
        info = {}
        return state, reward, done, info

    def reset(self):
        state = 0
        return state


env = BallSortEnv()