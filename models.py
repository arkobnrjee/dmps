import torch
import torch.nn as nn
from copy import deepcopy


HIDDEN_LAYER_SIZE = 256


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_LAYER_SIZE, device=device),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, device=device),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, act_dim, device=device),
            nn.Tanh()
        )

    def forward(self, s):
        return self.net(s)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, device, ret_low=-1., ret_high=1.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, HIDDEN_LAYER_SIZE, device=device),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, device=device),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYER_SIZE, 1, device=device)
        )

        self.scale = (ret_high - ret_low) / 2.
        self.center = (ret_high + ret_low) / 2.

    def forward(self, s, a):
        return self.scale * self.net(torch.cat((s, a), dim=-1)) + self.center


class Model(nn.Module):
    def __init__(self, obs_dim, act_dim, device, ret_low=-1., ret_high=1.):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim, device)
        self.critic1 = Critic(obs_dim, act_dim, device, ret_low=ret_low, ret_high=ret_high)
        self.critic2 = Critic(obs_dim, act_dim, device, ret_low=ret_low, ret_high=ret_high)
        self.actor_targ = deepcopy(self.actor)
        self.critic1_targ = deepcopy(self.critic1)
        self.critic2_targ = deepcopy(self.critic2)

    def critic(self, s, a):
        return torch.min(self.critic1(s, a), self.critic2(s, a))

    def critic_targ(self, s, a):
        return torch.min(self.critic1_targ(s, a), self.critic2_targ(s, a))
