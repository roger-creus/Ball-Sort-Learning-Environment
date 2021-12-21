import gym
import numpy as np
import os
import json
import colored
from colored import stylize
from stable_baselines.common.env_checker import check_env


class BallSortEnv(gym.Env):
    def __init__(self):
        """ A tube [1,2,3,4] is read from bottom to top """

        # Read level config from file
        self.f = open("levels/007-config.json")
        self.json_file = json.load(self.f)
        

        # Define color mapping for balls
        self.colors = {1: "red", 2: "blue", 3: "white", 4: "yellow", 5: "magenta", 6: "cyan", 7: "green", 8: "black", 9:"dark_green",  10:"light_green",  11:"light_yellow", 12:"dodger_blue_3"}
        

        # Define according to level config
        self.n_tubes = len(self.json_file["tubes"])
        self.n_colors = len(set([color for sublist in self.json_file["tubes"] for color in sublist]))
        self.n_actions = self.n_tubes**2

        self.state = np.zeros((self.n_tubes, 4))

        # Define action mapping 
        self.action_mapping = self.init_action_mapping()

        ## Define action and state spaces
        # Can make any move between a pair of tubes (although several might be ilegal)
        self.action_space = gym.spaces.Discrete(self.n_tubes**2)

        # Get a 3D matrix of shape (shape=(n_tubes, tube_length, n_colors))
        self.observation_space = gym.spaces.Box(low = np.array((1, 4)), high=np.array((self.n_tubes, 4), dtype = "float32"))

    def step(self, action):
        done = False

        source = self.action_mapping[action][0]
        goal = self.action_mapping[action][1]

        source_tube = self.state[source]
        goal_tube = self.state[goal]

        print("Action is from SOURCE: " + str(source) + " to GOAL: " + str(goal))     


        # get positions of the balls in the tube (0: bottom,..., 3: top)
        if np.nonzero(source_tube)[0].size != 0:
            source_ball_position = np.max(np.nonzero(source_tube))
            
        if np.nonzero(goal_tube)[0].size != 0:
            goal_ball_position = np.max(np.nonzero(goal_tube))
        else:
            goal_ball_position = -1

        ilegal = False


        ## ilegal actions 
        # (move to same tube)
        if source == goal:
            ilegal = True
            print("ilegal: moving to same tube")
        # (move to a full tube)
        elif goal_tube[-1] != 0:
            ilegal = True
            print("ilegal: moving to a full tube")
        # (move from an empty tube)
        elif np.nonzero(source_tube)[0].size == 0:
            ilegal = True
            print("ilegal: moving from an empty tube")
        # (move to tube with top color not same as source top color)
        elif source_tube[source_ball_position] != goal_tube[goal_ball_position] and np.nonzero(goal_tube)[0].size != 0:
            ilegal = True
            print("ilegal: moving to tube with not same color")
        
        if ilegal is not True:
            self.state[goal, goal_ball_position+1] = self.state[source, source_ball_position]
            self.state[source,source_ball_position] = 0
            reward = 1
        else:
            reward = -1

        # Always do this because for a legal move we overrite the self.state
        next_state = self.state
        
        if self.is_solved_state(next_state):
            done = True
            reward = 100
            print("SOLVED")
        elif self.is_terminal_state(next_state):
            done = True
            reward = -100
            print("TERMINAL")

        info = {}
   
        return next_state, reward, done, info

    def reset(self):
        self.state = np.zeros((self.n_tubes, 4))

        i = 0
        for tube in self.json_file["tubes"]:
            j = 0
            for ball in tube:
                self.state[i,j] = ball
                j += 1
            i += 1
        return self.state

    def render(self):
        scene = np.flip(np.transpose(self.state), axis = 0)

        for row in scene:
            for ball in row:
                print("|", end=' ')
                if ball != 0:
                    print(stylize("O", colored.fg(self.colors[ball])), end=' ')
                else:
                    print(" ", end=' ')
                    
                print("|", end=' ')
            print()
        print(self.n_tubes * "------")

    def init_action_mapping(self):
        mapping = {}
        x = 0
        for i in range(self.n_tubes):
            for j in range(self.n_tubes):
                mapping[x] = (i,j)
                x += 1

        return mapping
            

    def is_solved_state(self, state):
        for tube in state:
            if np.nonzero(tube)[0].size != 0:
                if len(set(tube)) != 1:
                    return False
        return True
                

    def is_terminal_state(self, state):
        i = 0
        
        for tube_i in state:
            # if there is an empty tube then state is not terminal
            if np.nonzero(tube_i)[0].size == 0:
                return False
            else:
                highest_ball_position_i = np.max(np.nonzero(tube_i))
                highest_ball_i = tube_i[highest_ball_position_i]
                j = 0
                for tube_j in state:
                    if i != j:
                        # if there is an empty tube then state is not terminal
                        if np.nonzero(tube_j)[0].size == 0:
                            return False
                        else:
                            highest_ball_position_j = np.max(np.nonzero(tube_j))
                            highest_ball_j = tube_j[highest_ball_position_j]

                            # if the top balls are same color in two different tubes and one of them is not full then state is not terminal
                            if highest_ball_i == highest_ball_j and (highest_ball_position_i != 3 or highest_ball_position_j != 3):
                                return False
                    j += 1
            i += 1
        return True
            


env = BallSortEnv()

check_env(env)

"""
env.reset()
env.render()
done = False
step = 0
while done is False:
    print("Step: " + str(step))
    next_state, reward, done, _ = env.step(env.action_space.sample()) # take a random action
    env.render()
    step += 1
env.close()
"""