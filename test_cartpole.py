from reinforce import Pi, train
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def main():
    env = gym.make('CartPole-v1')
    in_dim=env.observation_space.shape[0]
    out_dim = env.action_space.n
    pi = Pi(in_dim, out_dim)
    optimizer = optim.Adam(pi.parameters(), lr=0.01)
    for epi in range(400):
        state = env.reset()
        for t in range(200):
            action = pi.act(state)
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            env.render()
            if done:
                break
        loss = train(pi, optimizer)
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset()
        print(f'Episode {epi}, loss: {loss}, \
        total_reward: {total_reward}, solved:{solved}')
                   

if __name__ == '__main__':
    main()
