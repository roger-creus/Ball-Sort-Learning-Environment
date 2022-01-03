import gym_ballsort.envs
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import VecMonitor


env = gym.make("ballsort-v0")

model = PPO("MlpPolicy", env, verbose=0)
model_path = "models\ppo_ballsort_level_7.zip"
model = PPO.load(model_path)


obs = env.reset()
done = False
running_reward = 0
steps = 0
while True and done is False:
    steps += 1
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    running_reward += reward
    env.render()

print("in " + str(steps) + " steps")
