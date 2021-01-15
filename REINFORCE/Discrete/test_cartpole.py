from Policy import Pi, train
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import seaborn as sns
sns.set(style='darkgrid')

def generate_history(lr, gamma,env,MAX_EPISODES):
    
    in_dim=env.observation_space.shape[0]
    out_dim = env.action_space.n
    pi = Pi(in_dim, out_dim)
    rew_history = []
    
    optimizer = optim.Adam(pi.parameters(), lr=lr)

    print('GAMMA=',gamma)
    for epi in range(MAX_EPISODES):
        state = env.reset()
        for t in range(300):
            action = pi.act(state)
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            env.render()
            if done:
                break
        loss = train(pi, optimizer,gamma)
        total_reward = sum(pi.rewards)
        rew_history.append(total_reward)
        solved = total_reward > 295.0

        pi.onpolicy_reset()
        print(f'Episode {epi},total_reward {total_reward}, solved:{solved}')
    env.close()
    print()
    return rew_history

def main():
    env = gym.make('CartPole-v1')
    
    fig = plt.figure(figsize=(10,7))
    MAX_EPISODES = 150
    for g in [0.85,0.9,0.95,0.98]:
        rew_history = generate_history(0.01,g,env,MAX_EPISODES)
        sns.lineplot(x=[i for i in range(MAX_EPISODES)], y=rew_history,
                     label='γ='+str(g))
    plt.title('Total Reward History')
    plt.xlabel('Episode')
    plt.ylabel('Total Episode Reward')
    
    plt.show()
if __name__ == '__main__':
    main()
