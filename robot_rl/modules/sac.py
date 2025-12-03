import torch
import torch.nn as nn
from torch.distributions import Normal

from robot_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation

class Actor(nn.Module):
    def __init__(
        self,
        obs_dim,
        num_actions,
        hidden_dims=[256, 256, 256],
        activation="elu",
        max_action=1.0,
        last_activation: str | None = None
    ):
        super().__init__()

        self.max_action = max_action
        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None

        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            activation_mod,
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            activation_mod,
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            activation_mod
        )
        self.mean_layer = nn.Linear(hidden_dims[2], num_actions)
        self.log_std_layer = nn.Linear(hidden_dims[2], num_actions)

    def forward(self, obs):
        op = self.policy(obs)
        mu = self.mean_layer(op)
        # constraint log std for stability - do not make the distribution too wide
        log_std = torch.clamp(self.log_std_layer(op), -20, 2)
        std = torch.exp(log_std)

        # Sample action using reparameterization trick
        reparam_dist = Normal(mu, std)
        action = reparam_dist.rsample()
        
        # Squash action to [-1, 1] range using tanh, then scale to max_action
        squashed_action = torch.tanh(action) * self.max_action
        
        # Compute log probability with change of variables correction for tanh
        # log_prob = log π(u|s) - log|det(du/da)| where u = tanh(a)
        # log|det(du/da)| = sum(log(1 - tanh²(a)))
        log_prob = reparam_dist.log_prob(action).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - torch.tanh(action).pow(2) + 1e-6).sum(-1, keepdim=True)

        return squashed_action, log_prob

class Critic(nn.Module):
    """SAC Critic (Q-function) that takes state and action as input."""
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dims=[256, 256, 256],
        activation="elu",
        last_activation: str | None = None
    ):
        super().__init__()

        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None

        # Q-network takes concatenated state and action as input
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dims[0]),
            activation_mod,
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            activation_mod,
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            activation_mod,
            nn.Linear(hidden_dims[2], 1)
        )

        self.q_net2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dims[0]),
            activation_mod,
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            activation_mod,
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            activation_mod,
            nn.Linear(hidden_dims[2], 1)
        )

    def forward(self, obs, action):
        # Concatenate state and action
        q_input = torch.cat([obs, action], dim=-1)
        q_value = self.q_net(q_input)
        q_value2 = self.q_net2(q_input)
        return q_value, q_value2

class SACModule(nn.Module):
    def __init__(
        self,
        obs_dim,
        num_actions,
        max_action,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
    ):
        super().__init__()

        self.actor = Actor(
            obs_dim,
            num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            max_action=max_action
        )

        self.critic = Critic(
            obs_dim,
            num_actions,
            hidden_dims=critic_hidden_dims,
            activation=activation
        )

    def forward(self, obs, action):
        action, log_prob = self.actor(obs)
        q1, q2 = self.critic(obs, action)
        return action, log_prob, q1, q2