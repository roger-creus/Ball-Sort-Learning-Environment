import envs
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


#env = BallSortEnv()
#env = Monitor(env)

"""
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000, 
    "env_name": "ballsort-v0",
}
run = wandb.init(
    project="ballsort",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)
"""

env = make_vec_env("ballsort-v0", n_envs=4)

model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./ppo_tensorboard/")
model.learn(total_timesteps=1000000, tb_log_name="first_run")
#run.finish()

"""
reward_plot = plt.figure(1)
plt.plot(env.get_episode_rewards())
plt.title = "Episode Rewards"

length_plot = plt.figure(2)
plt.plot(env.get_episode_lengths())
plt.title = "Episode Lengths"

plt.show()
"""

model.save("ppo_ballsort")

test_env = BallSortEnv()
obs = test_env.reset()
done = False

while done is False:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = test_env.step(action)
    test_env.render()

