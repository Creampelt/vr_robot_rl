from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from robot_rl.modules import ActorCritic
from robot_rl.networks import MLP, EmpiricalNormalization


class ActorCriticEstimator(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict,
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        estimator_index: int = -1,
        oracle: bool = False,
        estimate_obs: bool = False,
        estimate_next_obs: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticEstimator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization,
            critic_obs_normalization,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
            noise_std_type,
        )

        self.estimate_obs = estimate_obs
        self.estimate_next_obs = estimate_next_obs

        # get the observation dimensions
        self.num_estimates: int = 0
        for obs_group in obs_groups["estimator"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticEstimator module only supports 1D observations."
            self.num_estimates += obs[obs_group].shape[-1]
        if estimate_obs:
            self.num_estimates += self.num_actor_obs
        if estimate_next_obs:
            self.num_estimates += self.num_actor_obs

        # backbone
        shared_hidden_dims = actor_hidden_dims[:estimator_index]
        backbone_output_dim = shared_hidden_dims[-1]
        self.backbone = MLP(self.num_actor_obs, backbone_output_dim, shared_hidden_dims[:-1], activation)

        # estimator
        estimator_dims = actor_hidden_dims if oracle else shared_hidden_dims
        self.estimator = MLP(self.num_actor_obs, self.num_estimates, estimator_dims, activation)
        # estimator observation normalization
        self.estimator_obs_normalizer = EmpiricalNormalization(self.num_estimates)
        print(f"Estimator MLP: {self.estimator}")

    def get_estimator_obs(self, obs: TensorDict, last_obs: TensorDict | None = None, **kwargs) -> torch.Tensor:
        if last_obs is not None:
            next_obs = obs
            obs = last_obs
        obs_list = []
        for obs_group in self.obs_groups["estimator"]:
            obs_list.append(obs[obs_group])
        if self.estimate_obs:
            obs_list.append(self.get_actor_obs(obs))
        if self.estimate_next_obs:
            obs_list.append(self.get_actor_obs(next_obs))
        return torch.cat(obs_list, dim=-1)

    def update_normalization(self, obs: TensorDict, last_obs: torch.Tensor | None = None, **kwargs) -> None:
        super().update_normalization(obs)
        estimator_obs = self.get_estimator_obs(obs, last_obs)
        self.estimator_obs_normalizer.update(estimator_obs)

    def estimate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.estimator(obs)

    def estimate_inference(self, obs: TensorDict, unnorm: bool = True) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        if not unnorm:
            obs = self.actor_obs_normalizer(obs)
        return self.estimator(obs)

    def get_latents(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        return self.backbone(obs)

    def get_last_layer(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # Get the weight matrix of the last layer of the actor
        actor_last_layer = self.actor[-1]
        actor_weights = actor_last_layer.weight if isinstance(actor_last_layer, nn.Linear) else None

        # Get the weight matrix of the last layer of the estimator
        estimator_last_layer = self.estimator[-1]
        estimator_weights = estimator_last_layer.weight if isinstance(estimator_last_layer, nn.Linear) else None

        return actor_weights, estimator_weights


def resolve_estimator_config(alg_cfg: dict) -> tuple[dict, bool]:
    """Resolve the estimator configuration.

    Args:
        alg_cfg: The algorithm configuration dictionary.

    Returns:
        A tuple containing the resolved algorithm configuration dictionary and whether the last observations
        should be used (i.e. if estimate_next_obs is True)
    """
    alg_cfg["estimator"] = bool(alg_cfg.get("estimator_cfg") is not None)
    use_last_obs = bool(alg_cfg["estimator"] and alg_cfg["estimator_cfg"]["estimate_next_obs"])
    return alg_cfg, use_last_obs
