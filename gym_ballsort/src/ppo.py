import gym_ballsort.envs
import gym
import numpy as np
from gym_ballsort.envs.ballsort import BallSortEnv
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.transforms import ToTensor, Lambda, Normalize
from torchvision import transforms
import wandb
import random
import os
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self, obs_len, act_len):
        super(Agent, self).__init__()
        
        self.obs_len = obs_len
        self.act_len = act_len

        self.mlp = nn.Sequential(
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128, act_len)
        self.critic = nn.Linear(128, 1)


    def forward(self, state):
        state = torch.nn.functional.normalize(torch.flatten(state, start_dim = 1))
        
        out = self.mlp(state)
        
        action_scores = self.actor(out)
        state_value = self.critic(out)
        return F.softmax(action_scores, dim=1), state_value

    def compute_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self(state)

        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        return action.item(), m.log_prob(action).item(), state_value.item()


transition = np.dtype([('s', np.float64, (9,4)), ('a', np.float64), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (9,4))])

class ReplayMemory():
    def __init__(self, capacity):
        self.buffer_capacity = capacity
        self.buffer = np.empty(capacity, dtype=transition)
        self.counter = 0

    # Stores a transition and returns True or False depending on whether the buffer is full or not
    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

def train(policy, optimizer, memory, hparams):

    gamma = hparams['gamma']
    ppo_epoch = hparams['ppo_epoch']
    batch_size = hparams['batch_size']
    clip_param = hparams['clip_param']
    c1 = hparams['c1']
    c2 = hparams['c2']


    s = torch.tensor(memory.buffer['s'], dtype=torch.float)
    a = torch.tensor(memory.buffer['a'], dtype=torch.float)
    r = torch.tensor(memory.buffer['r'], dtype=torch.float).view(-1, 1)
    s_ = torch.tensor(memory.buffer['s_'], dtype=torch.float)

    old_a_logp = torch.tensor(memory.buffer['a_logp'], dtype=torch.float).view(-1, 1)

    with torch.no_grad():
        target_v = r + gamma * policy(s_)[1]
        adv = target_v - policy(s)[1]

    for _ in range(ppo_epoch):
        for index in BatchSampler(SubsetRandomSampler(range(memory.buffer_capacity)), batch_size, False):
            probs, _ = policy(s[index])
            dist = Categorical(probs)
            entropy = dist.entropy()
            
            a_logp = dist.log_prob(a[index]).unsqueeze(dim=1)

            ratio = torch.exp(a_logp - old_a_logp[index])

            surr1 = ratio * adv[index]
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv[index]

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.smooth_l1_loss(policy(s[index])[1], target_v[index])
            entropy = - entropy.mean()
 
            loss = policy_loss + c1 * value_loss + c2 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    return policy_loss.item(), value_loss.item(), -entropy.item(), ratio.mean().item()

def test(env, policy, render=False):
    state, ep_reward, done = env.reset(), 0, False
    game_steps = 0
    while not done and game_steps < hparams["max_episode_length"]:
        game_steps += 1
        env.render()
        action, _, _ = policy.compute_action(state)
        state, reward, done, _ = env.step(action)
        ep_reward += reward

    env.close()
    return ep_reward

hparams = {
    'gamma' : 0.99,
    'log_interval' : 100,
    'max_episode_length' : 500,
    'num_episodes': 10000,
    'lr' : 0.0003,
    'clip_param': 0.2,
    'ppo_epoch': 10,
    'replay_size': 500,
    'batch_size': 64,
    'c1': 0.5,
    'c2': 0.01
}


env = BallSortEnv()
seed=0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Get number of actions from gym action space
n_inputs = env.observation_space.shape[0]
n_actions = env.action_space.n

# Initialize wandb run
wandb.finish() # execute to avoid overlapping runnings (advice: later remove duplicates in wandb)
wandb.init(project="ballsort", config=hparams)
wandb.run.name = 'ppo_ballsort'


# Create policy and optimizer
policy = Agent(n_inputs, n_actions)
optimizer = torch.optim.Adam(policy.parameters(), lr=hparams['lr'])

eps = np.finfo(np.float32).eps.item()
memory = ReplayMemory(hparams['replay_size'])

# Training loop
running_reward = 0
ep_rew_history_reinforce = []
for i_episode in range(hparams['num_episodes']):
    # Collect experience
    state, ep_reward, done = env.reset(), 0, False
    game_steps = 0
    while not done and game_steps < hparams["max_episode_length"]:  # Don't infinite loop while learning
        game_steps += 1
        action, a_logp, state_value = policy.compute_action(state)
        next_state, reward, done, _ = env.step(action)
        

        if memory.store((state, action, a_logp, reward, next_state)):
            policy_loss, value_loss, avg_entropy, ratio = train(policy, optimizer, memory, hparams)
            wandb.log(
                {
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'running_reward': running_reward,
                'mean_entropy': avg_entropy,
                'ratio': ratio
                })


        state = next_state

        ep_reward += reward
        if done:
            break

    # Update running reward
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    
    
    ep_rew_history_reinforce.append((i_episode, ep_reward))
    if i_episode % hparams['log_interval'] == 0:
        print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')
        test_env = BallSortEnv()
        ep_reward = test(test_env, policy)

print(f"Finished training! Running reward is now {running_reward}")

wandb.finish()