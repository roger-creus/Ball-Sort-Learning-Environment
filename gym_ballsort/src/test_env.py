import gym_ballsort.envs
import gym
from stable_baselines3 import PPO
#from stable_baselines3 import DQN
#from stable_baselines3 import A2C



#model = PPO("MlpPolicy", env, verbose=0)
model_path = "models\ppo_ballsort_level_7.zip"
model = PPO.load(model_path)

env = gym.make("ballsort-v0")
obs = env.reset()
done = False
steps = 0
running_reward = 0

while True and done is False:
    env.render()
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    running_reward += reward

env.close()
print("reward: " + str(running_reward))
