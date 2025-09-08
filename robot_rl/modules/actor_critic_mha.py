from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal
from tensordict import TensorDict

from robot_rl.utils import resolve_nn_activation
from robot_rl.networks import MLP, EmpiricalNormalization


class MHAEncoder(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_latent: int,
        n_heads: int,
        n_channels: int,
        kernel_size: int,
        dropout: float = 0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.n_latent = n_latent
        self.proprio_enc = nn.Sequential(
            nn.Linear(n_obs, n_latent),
            resolve_nn_activation(activation),
        )
        self.height_enc = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size, padding="same"),
            resolve_nn_activation(activation),
            nn.Conv2d(n_channels, n_latent - 3, kernel_size, padding="same"),
            resolve_nn_activation(activation),
        )
        self.mha = nn.MultiheadAttention(n_latent, n_heads, dropout, batch_first=True)

    def map_enc(self, map_obs: torch.Tensor) -> torch.Tensor:
        b, l, w, _ = map_obs.shape
        map_z = map_obs[..., 2].unsqueeze(dim=1)  # [b, 1, l, w]
        z_height = self.height_enc(map_z)  # [b, d-3, l, w]
        z_map = torch.cat((map_obs.view(b, 3, l, w), z_height), dim=1)  # [b, d, l, w]
        return z_map.view(b, l * w, -1)  # [b, l * w, d]

    def forward(
        self, proprio_obs: torch.Tensor, map_obs: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        z_map = self.map_enc(map_obs)  # [b, l * w, d]
        z_proprio = self.proprio_enc(proprio_obs).unsqueeze(dim=1)  # [b, 1, d]
        z_mha, weights = self.mha(z_proprio, z_map, z_map, need_weights=need_weights)
        z = torch.cat((z_mha.squeeze(dim=1), proprio_obs), dim=-1)  # [b, d + d_obs]

        return z, (weights if need_weights else None)


class ActorCriticMHA(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict,
        num_actions: int,
        n_latent: int,
        n_heads: int,
        n_channels: int,
        kernel_size: int,
        n_rows: int,
        n_cols: int,
        dropout: float = 0,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="relu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticMHA.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs: int = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs: int = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        assert num_actor_obs == num_critic_obs, (
            f"Expected actor and critic obs to be the same size, instead got {num_actor_obs=} and {num_critic_obs=}."
        )

        d_proprio = num_actor_obs - (n_rows * n_cols * 3)
        d_enc = n_latent + d_proprio

        self.encoder = MHAEncoder(d_proprio, n_latent, n_heads, n_channels, kernel_size, dropout, activation)
        print(f"MHA Encoder: {self.encoder}")

        # actor
        self.actor_head = MLP(d_enc, num_actions, actor_hidden_dims, activation)
        # actor observation normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # critic
        self.critic_head = MLP(d_enc, 1, critic_hidden_dims, activation)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(self.num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic_head}")

        self.l = n_rows
        self.w = n_cols

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        # attention weight heatmap
        self.weights = None

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_attention_map(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        proprio_obs = obs[:, : -self.w * self.l * 3]
        map_obs = obs[:, -self.w * self.l * 3 :].view(-1, self.l, self.w, 3)
        _, weights = self.encoder(proprio_obs, map_obs, need_weights=True)
        return weights

    def update_distribution(self, obs: torch.Tensor) -> None:
        # compute mean
        proprio_obs = obs[:, : -self.w * self.l * 3]
        map_obs = obs[:, -self.w * self.l * 3 :].view(-1, self.l, self.w, 3)
        z, self.weights = self.encoder(proprio_obs, map_obs, need_weights=False)
        mean = self.actor_head(z)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        proprio_obs = obs[:, : -self.w * self.l * 3]
        map_obs = obs[:, -self.w * self.l * 3 :].view(-1, self.l, self.w, 3)
        z, _ = self.encoder(proprio_obs, map_obs, need_weights=False)
        actions_mean = self.actor_head(z)
        return actions_mean

    def act_inference_vis(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        proprio_obs = obs[:, : -self.w * self.l * 3]
        map_obs = obs[:, -self.w * self.l * 3 :].view(-1, self.l, self.w, 3)
        z, weights = self.encoder(proprio_obs, map_obs, need_weights=True)
        actions_mean = self.actor_head(z)
        return actions_mean, weights

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        proprio_obs = obs[:, : -self.w * self.l * 3]
        map_obs = obs[:, -self.w * self.l * 3 :].view(-1, self.l, self.w, 3)
        z, _ = self.encoder(proprio_obs, map_obs, need_weights=False)
        value = self.critic_head(z)
        return value

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True


if __name__ == "__main__":
    b = 4096
    num_actor_obs = 48
    n_rows = 20
    n_cols = 20

    obs = TensorDict({"proprio": torch.randn(b, num_actor_obs), "map": torch.randn(b, n_rows, n_cols, 3)})
    obs_groups = {"policy": ["proprio", "map"], "critic": ["proprio", "map"]}

    policy = ActorCriticMHA(
        obs,
        obs_groups,
        num_actions=12,
        n_latent=64,
        n_heads=16,
        n_channels=16,
        kernel_size=5,
        n_rows=n_rows,
        n_cols=n_cols,
        dropout=0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="relu",
        init_noise_std=1.0,
        noise_std_type="scalar",
    )

    out = policy.act(obs)
    weights = policy.get_attention_map(obs)

    print(f"Proprio Obs: {obs['proprio'].shape}")
    print(f"Map Obs: {obs['map'].shape}")
    print(f"Actions: {out.shape}")
    print(f"Weights: {weights.shape}")
