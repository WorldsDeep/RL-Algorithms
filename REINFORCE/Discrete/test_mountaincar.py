from reinforce import Pi, train
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')

def generate_history(lr, gamma,env,MAX_EPISODES, display=False):
    
    in_dim=env.observation_space.shape[0]
    out_dim = env.action_space.n
    pi = Pi(in_dim, out_dim)
    loss_history = []
    
    optimizer = optim.Adam(pi.parameters(), lr=lr)
    print('GAMMA=',gamma)
    for epi in range(MAX_EPISODES):
        state = env.reset()
        for t in range(500):
            action = pi.act(state)
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            if display:
                env.render()
            if done:
                break
        loss = train(pi, optimizer,gamma)
        total_reward = sum(pi.rewards)
        loss_history.append(loss)
        solved = total_reward >-200
        pi.onpolicy_reset()
        print(f'Episode {epi},total_reward {total_reward}, loss {loss}, solved:{solved}')
    env.close()
    print()
    return loss_history

if __name__=='__main__':
    env = gym.make('MountainCar-v0')
    ffts = []
    
    fig = plt.figure()
    MAX_EPISODES = 250
    lr = 0.005
    for g in [0.75,0.8,0.85,0.9]:
        loss_history = generate_history(0.01,g,env,MAX_EPISODES)
        sns.lineplot(x=[i for i in range(MAX_EPISODES)], y=loss_history,
                     label='gamma='+str(g))
    plt.title('Total Reward History')
    plt.xlabel('Episode')
    plt.ylabel('Total Episode Reward')
    
    plt.show()
    
