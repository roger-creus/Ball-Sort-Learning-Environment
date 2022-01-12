import gym_ballsort.envs
from numpy.core.fromnumeric import mean
import gym
import numpy as np
from gym_ballsort.envs.ballsort import BallSortEnv
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback
from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import VecMonitor


env = make_vec_env("ballsort-v0", n_envs=1)

model = DQN("MlpPolicy", env, verbose=0, tensorboard_log="./dqn_tensorboard/")
model.learn(total_timesteps=5000000, tb_log_name="level 7 tube reward")
model.save("dqn_ballsort_level_7")
