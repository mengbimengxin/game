# coding: utf-8

import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from game_v3 import Environment, Game
from network_model import PolicyNetwork, ValueNetwork

from IPython.display import clear_output
import matplotlib.pyplot as plt
import numba

# get_ipython().run_line_magic('matplotlib', 'inline')


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def _reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=50000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


def plot(frame_idx, rewards, success_rates=None):
    # clear_output(True)
    plt.figure(figsize=(5, 5))
    plt.subplot(121)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.xlabel("frames")
    plt.ylabel("reward")
    plt.plot(rewards)
    if success_rates is not None:
        plt.subplot(122)
        plt.title("成功率:{:.3f}".format(success_rates[-1]))
        plt.xlabel("episodes")
        plt.ylabel("success rate")
        plt.plot(success_rates)
    plt.show()


@numba.jit
def ddpg_update(batch_size,
                gamma=0.99,
                min_value=-np.inf,
                max_value=np.inf,
                soft_tau=1e-2):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    policy_loss = value_net(state, policy_net(state))
    policy_loss = -policy_loss.mean()

    next_action = target_policy_net(next_state)
    target_value = target_value_net(next_state, next_action.detach())
    expected_value = reward + (1.0 - done) * gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = value_net(state, action)
    value_loss = value_criterion(value, expected_value.detach())

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


        # In[10]:


# @numba.jit
def train():
    frame_idx = 0
    episode = 1
    steps = []
    successed_rate = 0
    successed_rates = []
    while frame_idx < max_frames:
        state = env.reset()
        ou_noise.reset()
        episode_reward = 0
        steps.append(0)
        for step in range(max_steps):
            action = policy_net.get_action(state)
            action = ou_noise.get_action(action, step)
            next_state, reward, done, n = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            if len(replay_buffer) > 3 * batch_size:
                ddpg_update(batch_size)

            state = next_state
            episode_reward += reward
            frame_idx += 1
            steps[-1] += 1

            if done:
                successed_rate += 1
                break
            if n > 0:
                break

        if episode % 10 == 0:
            print("回合: {}\n".format(str(episode)),
                  "帧数: {}\n".format(str(frame_idx)),
                  "每回合平均得分: {:.2f}\n".format(rewards[-1] / steps[-1]),
                  "每回合执行步数: {:.2f}\n".format(sum(steps) / episode),
                  "成功率: {:.2f}\n".format(successed_rate / episode))
            successed_rates.append([episode, successed_rate / episode])

        episode += 1

        rewards.append(episode_reward)

    torch.save(target_policy_net, './model_save/target_policy_net.pkl')
    torch.save(target_value_net, './model_save/target_value_net.pkl')
    try:
        for j, this_r in enumerate(rewards):
            if j > 0:
                rewards[j] = 0.9 * rewards[j - 1] + 0.1 * rewards[j]
        for j, this_r in enumerate(successed_rates):
            if j > 0:
                successed_rates[j][1] = 0.9 * successed_rates[j - 1][1] + 0.1 * successed_rates[j][1]

        plot(frame_idx, rewards, successed_rates)
    except Exception:
        pass


if __name__ == '__main__':
    env = Game(Environment(window_size=(250, 250), obstructs=2))
    ou_noise = OUNoise(env.action_space)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 256

    value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(param.data)

    value_lr = 1e-3
    policy_lr = 1e-4

    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
    # value_optimizer = optim.Adadelta(value_net.parameters(), lr=value_lr)
    # policy_optimizer = optim.Adadelta(policy_net.parameters(), lr=policy_lr)

    value_criterion = nn.MSELoss()

    replay_buffer_size = 10000
    replay_buffer = ReplayBuffer(replay_buffer_size)

    max_frames = 22000
    max_steps = 150
    rewards = []
    batch_size = 256

    train()
