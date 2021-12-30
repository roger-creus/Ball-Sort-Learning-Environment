from gym.envs.registration import register 
import gym

register(id='ballsort-v0',entry_point='gym_ballsort.envs:BallSortEnv') 

env = gym.make('ballsort-v0')