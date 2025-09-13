from __future__ import annotations

from typing import Callable, Literal, cast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.hooks import RemovableHandle
from tensordict import TensorDict

from robot_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticEstimator, Probe, SAE
from robot_rl.networks import EmpiricalNormalization


class ProbeAlg:
    policy: ActorCritic | ActorCriticRecurrent | ActorCriticEstimator

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent | ActorCriticEstimator,
        probes: dict[str, nn.Module],
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_learning_epochs: int = 1,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        oracle: bool = False,
        probe_type: Literal["linear", "sae"] = "linear",
        probe_obs: bool = False,
        probe_next_obs: bool = False,
        **kwargs,
    ) -> None:
        self.device = device
        self.policy = policy.to(self.device)
        self.probes = {k: v.to(self.device) for k, v in probes.items()}
        self.probe_type = probe_type
        self.learning_rate = learning_rate
        self.num_learning_epochs = num_learning_epochs
        self.obs_groups = obs_groups

        # store probe config
        # self.probe_type: Literal["sae", "linear"] = probe_cfg["class_name"].lower()
        # if self.probe_type not in ["sae", "linear"]:
        #     raise ValueError(
        #         f"Received invalid probe type. Expected one of ['sae', 'linear'], but got {self.probe_type}."
        #     )
        self.probe_obs: bool = self.probe_type == "linear" and probe_obs
        self.probe_next_obs: bool = self.probe_type == "linear" and probe_next_obs
        self.oracle: bool = oracle

        # compute probe slices if necessary
        self.probe_slices: list[slice] | None = None
        if self.probe_type == "linear":
            # get obs size
            num_probe_obs = 0
            for obs_group in obs_groups["probe"]:
                assert len(obs[obs_group].shape) == 2, "The ProbeAlg module only supports 1D observations."
                num_probe_obs += obs[obs_group].shape[-1]
            num_actor_obs = 0
            for obs_group in obs_groups["policy"]:
                assert len(obs[obs_group].shape) == 2, "The ProbeAlg module only supports 1D observations."
                num_actor_obs += obs[obs_group].shape[-1]

            self.probe_slices = [slice(0, num_probe_obs)]
            if self.probe_obs:
                self.probe_slices.append(slice(num_probe_obs, num_probe_obs + num_actor_obs))
                num_probe_obs += num_actor_obs
            if self.probe_next_obs:
                self.probe_slices.append(slice(num_probe_obs, num_probe_obs + num_actor_obs))
                num_probe_obs += num_actor_obs

            # create probe normalizer
            self.probe_obs_normalizer = EmpiricalNormalization(shape=(num_probe_obs,), until=1.0e8).to(self.device)

        params: list[nn.Parameter] = []
        for v in self.probes.values():
            params += [p for p in v.parameters()]
        self.optimizer = optim.Adam(params, lr=learning_rate)

        self.latents: dict[str, torch.Tensor] = {}
        self.hooks: dict[str, RemovableHandle] = {}

        def get_latents(name: str) -> Callable:
            def hook(module, input, output) -> None:
                self.latents[name] = output.detach()

            return hook

        actor_module = cast(nn.Module, self.policy.estimator if self.oracle else self.policy.actor)
        for name, module in actor_module.named_children():
            self.hooks[name] = module.register_forward_hook(get_latents(name))

    def test_mode(self) -> None:
        self.policy.test()

    # def train_mode(self):
    #    self.policy.train()

    def get_probe_obs(self, obs: TensorDict, last_obs: TensorDict | None = None) -> torch.Tensor:
        if last_obs is not None:
            next_obs = obs
            obs = last_obs
        obs_list = []
        for obs_group in self.obs_groups["probe"]:
            obs_list.append(obs[obs_group])
        if self.probe_obs:
            obs_list.append(self.policy.get_actor_obs(obs))
        if self.probe_next_obs:
            obs_list.append(self.policy.get_actor_obs(next_obs))
        return torch.cat(obs_list, dim=-1)

    def act(self, obs: TensorDict) -> torch.Tensor:
        return self.policy.act_inference(obs)

    def estimate(self, obs: TensorDict) -> torch.tensor:
        if not isinstance(self.policy, ActorCriticEstimator):
            raise ValueError(f"Policy class {type(self.policy)} does not have an estimator.")
        return self.policy.estimate(obs)

    def extract_latents(self) -> dict[str, torch.Tensor]:
        return self.latents

    def probe_inference(self, obs: TensorDict, unnorm: bool = False) -> torch.Tensor:
        self.act(obs)
        latents = self.extract_latents()
        probe_inf = {k: v(latents[k]) for k, v in self.probes.items()}
        if unnorm and self.probe_type == "linear":
            probe_inf = {k: self.probe_obs_normalizer.inverse(v) for k, v in probe_inf.items()}
        return probe_inf

    def update(
        self,
        latents: dict[str, torch.Tensor],
        dones: torch.Tensor,
        obs: TensorDict | None,
        last_obs: TensorDict | None = None,
        # probe_idx: list[int] = [],
        names: list[str] = ["Probe", "Obs", "Next_Obs"],
    ) -> tuple[dict[str, float], dict[str, dict[str, torch.Tensor]]]:
        loss_dict: dict[str, float] = {}
        error_dict: dict[str, dict[str, torch.Tensor]] = {}

        for i in range(self.num_learning_epochs):
            loss: torch.Tensor = torch.tensor(0).to(self.device)
            if self.probe_type == "sae":
                sparsity_loss = torch.tensor(0).to(self.device)
            for k, v in self.probes.items():
                if self.probe_type == "linear":
                    estimate = v(latents[k].clone())
                    target = self.get_probe_obs(obs, last_obs)
                    target = self.probe_obs_normalizer(target)
                else:
                    target = latents[k].clone()
                    estimate, sparsity = v(target)
                    sparsity_loss += sparsity.mean()
                se = (target - estimate).pow(2)
                se = torch.where(dones, 0, se)  # no loss on reset
                loss += se.mean()

                if self.probe_type == "linear" and i == self.num_learning_epochs - 1:
                    assert self.probe_slices is not None
                    for name, s in zip(names, self.probe_slices, strict=True):
                        error_dict[k][name] = se[:, s].pow(0.5).mean(dim=0)

            if self.probe_type == "sae":
                loss += sparsity_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_dict["loss"] = loss.item()
        loss_dict["mse"] = loss.item()
        if self.probe_type == "sae":
            loss_dict["sparsity"] = sparsity_loss.item()
            loss_dict["mse"] -= sparsity_loss.item()

        return loss_dict, error_dict
