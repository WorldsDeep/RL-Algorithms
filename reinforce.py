import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym



class Pi(nn.Module):
    """
    Neural Network Policy for REINFORCE algorithm.
    
    """
    def __init__(self, in_dim, out_dim, layer_size = 64):
    """
    Args:
      in_dim(int) :  Size of observation space
      out_dim(int):  Size of action space
      
    """
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, out_dim)
        ]

        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()
   
    # Reset Policy history
    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
    

    def forward(self, x):
        """
        Forward Propogation
          Args:
            x: torch.tensor of size self.in_dim
          Returns:
            pdparam: torch.tensor of size self.out_dim
        """
        pdparam = self.model(x)
        return pdparam
    


    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        pdparam = self.forward(x)
        pd = Categorical(logits = pdparam)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()
    

def train(pi, optimizer, gamma=0.99):
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs*rets
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
