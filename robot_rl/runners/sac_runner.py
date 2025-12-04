from __future__ import annotations

from typing import Callable

import os
import statistics
import time
import torch
import warnings
import numpy as np
from collections import deque
from tensordict import TensorDict

import robot_rl
from robot_rl.env import VecEnv
from robot_rl.utils import resolve_obs_groups, store_code_state

class SACRunner:
    """SAC runner for VOILA navigation training with world model expert frames."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        # SAC config has flat structure, not nested under "algorithm"
        self.alg_cfg = train_cfg.get("algorithm", train_cfg)  # Fallback to train_cfg if no algorithm key
        self.policy_cfg = train_cfg.get("policy", {})
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # store training configuration
        num_steps = self.cfg.get("num_steps_per_env", 100)
        # Handle case where num_steps_per_env might be nested or a dict
        if isinstance(num_steps, dict):
            print(f"[WARNING] num_steps_per_env is a dict: {num_steps}, using default value 100")
            self.num_steps_per_env = 100
        else:
            self.num_steps_per_env = num_steps
        self.save_interval = self.cfg["save_interval"]
        
        # SAC-specific configuration
        self.replay_buffer_size = self.cfg.get("replay_buffer_size", 100000)
        self.batch_size = self.cfg.get("batch_size", 256)
        self.warmup_steps = self.cfg.get("warmup_steps", 1000)
        self.update_frequency = self.cfg.get("update_frequency", 1)
        self.gradient_steps = self.cfg.get("gradient_steps", 1)  # Number of gradient steps per update

        # query observations from environment for algorithm construction
        obs_td = self.env.get_observations()
        default_sets = ["critic"]
        self.cfg["obs_groups"] = resolve_obs_groups(obs_td, self.cfg.get("obs_groups", {}), default_sets)

        # select observation tensor for policy: prefer camera latents if present, else policy group
        init_obs = obs_td.get("camera", obs_td.get("policy"))
        if init_obs is None:
            raise RuntimeError("No compatible observation group found: expected 'camera' or 'policy'.")

        # create the algorithm (SAC actor-critic) using dynamic state dimension
        self.alg = self._construct_algorithm(init_obs)
        
        # SAC-specific parameters
        self.gamma = self.alg_cfg.get("gamma", 0.99)
        self.tau = self.alg_cfg.get("tau", 0.005)
        # Prefer env.action_space.high for max action if available, else fallback to policy config
        env_action_space = getattr(self.env, "action_space", None)
        if env_action_space is not None and hasattr(env_action_space, "high"):
            try:
                import numpy as np
                high = env_action_space.high
                self.max_action = float(high) if np.isscalar(high) else float(np.max(high))
            except Exception:
                self.max_action = self.policy_cfg.get("max_action", 1.0)
        else:
            self.max_action = self.policy_cfg.get("max_action", 1.0)
        # Respect typical SAC default: target entropy = -action_dim
        self.target_entropy = self.alg_cfg.get("target_entropy", -self.env.num_actions)
        
        # Automatic temperature tuning
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alg_cfg.get("learning_rate", 3e-4))

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [robot_rl.__file__]
        
        # VOILA-specific modules
        self.expert_descriptors = []

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self._prepare_logging_writer()

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        
        # Initialize state tracking
        num_envs = self.env.num_envs
        action_dim = self.env.num_actions
        
        # Determine observation dimension dynamically from initial obs
        obs_dict_initial = self.env.get_observations().to(self.device)
        current_obs = obs_dict_initial.get("camera", obs_dict_initial.get("policy"))
        if current_obs is None:
            raise RuntimeError("No compatible observation group found at learn start: expected 'camera' or 'policy'.")
        obs_dim = current_obs.shape[-1]

        # Previous latents (z_{t-1}) - initialize with zeros
        prev_latent = torch.zeros((num_envs, obs_dim), device=self.device)
        # Previous actions (a_{t-1})
        prev_action = torch.zeros((num_envs, action_dim), device=self.device)
        # Expert frame indices for each environment
        expert_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        
        # Get initial observations (latents) from environment
        obs_dict = self.env.get_observations().to(self.device)
        # Phase 1: Initial latent/obs (prefer camera, else policy)
        current_latent = obs_dict.get("camera", obs_dict.get("policy"))  # z_t (num_envs, D)
        if current_latent is None:
            raise RuntimeError("No compatible observation group found during rollout: expected 'camera' or 'policy'.")
        
        # Construct initial state dynamically: [z_t, z_{t-1}, a_{t-1}]
        state = torch.cat([current_latent, prev_latent, prev_action], dim=-1)
        
        # Switch to train mode
        self.train_mode()
        
        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        
        # Efficient circular replay buffer using numpy arrays
        state_dim = obs_dim * 2 + action_dim
        replay_buffer = {
            "states": np.zeros((self.replay_buffer_size, state_dim), dtype=np.float32),
            "actions": np.zeros((self.replay_buffer_size, action_dim), dtype=np.float32),
            "rewards": np.zeros((self.replay_buffer_size, 1), dtype=np.float32),
            "next_states": np.zeros((self.replay_buffer_size, state_dim), dtype=np.float32),
            "dones": np.zeros((self.replay_buffer_size, 1), dtype=np.float32)
        }
        buffer_pos = 0
        buffer_full = False
        
        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            if hasattr(self.alg, "broadcast_parameters"):
                self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        global_step = 0
        
        for it in range(start_iter, tot_iter):
            start = time.time()
            
            # Phase 2: Action Selection (Actor)
            with torch.no_grad():
                if global_step < self.warmup_steps:
                    # Random exploration during warmup (scaled by max_action)
                    actions = (torch.rand((num_envs, action_dim), device=self.device) * 2 - 1) * self.max_action
                else:
                    # Sample from actor policy
                    actions, _ = self.alg.actor(state)
            
            # Phase 3: Environment Response
            obs_dict, rewards, dones, extras = self.env.step(actions.to(self.env.device))
            obs_dict, rewards, dones = (obs_dict.to(self.device), rewards.to(self.device), dones.to(self.device))
            latents_next = obs_dict.get("camera", obs_dict.get("policy"))
            
            # Next latent directly from env observations
            next_latent = latents_next  # z_{t+1}
            
            # Phase 4: VOILA Reward Calculation (now handled by env.step)
            # voila_rewards = self._compute_voila_rewards(
            #     camera_obs_next, actions, expert_indices, dones, extras, num_expert_frames
            # )
            voila_rewards = rewards  # Use rewards from environment
            
            # Construct next state
            next_state = torch.cat([next_latent, current_latent, actions], dim=-1)
            
            # Phase 5: Learning (SAC Update)
            # Store transitions in replay buffer (vectorized for efficiency)
            for env_idx in range(num_envs):
                replay_buffer["states"][buffer_pos] = state[env_idx].cpu().numpy()
                replay_buffer["actions"][buffer_pos] = actions[env_idx].cpu().numpy()
                replay_buffer["rewards"][buffer_pos] = voila_rewards[env_idx].cpu().numpy()
                replay_buffer["next_states"][buffer_pos] = next_state[env_idx].cpu().numpy()
                replay_buffer["dones"][buffer_pos] = dones[env_idx].cpu().numpy()
                
                buffer_pos = (buffer_pos + 1) % self.replay_buffer_size
                if buffer_pos == 0:
                    buffer_full = True
            
            # SAC gradient updates with batch collection
            loss_dict = {}
            if global_step >= self.warmup_steps and global_step % self.update_frequency == 0:
                buffer_size = self.replay_buffer_size if buffer_full else buffer_pos
                if buffer_size >= self.batch_size:
                    # Multiple gradient steps per update (batch collection pattern)
                    loss_dict = self._sac_update(replay_buffer, buffer_size)
            
            stop = time.time()
            collection_time = stop - start
            learn_time = 0.0  # SAC updates are async
            
            # Phase 6: Reset for Next Step
            # Update state tracking
            prev_latent = current_latent.clone()
            prev_action = actions.clone()
            state = next_state.clone()
            current_latent = next_latent.clone()
            
            # Handle episode resets
            for env_idx in range(num_envs):
                cur_reward_sum[env_idx] += voila_rewards[env_idx]
                cur_episode_length[env_idx] += 1
                
                if dones[env_idx]:
                    # Log episode info
                    if "episode" in extras and extras["episode"] is not None:
                        ep_infos.append(extras["episode"][env_idx] if isinstance(extras["episode"], (list, torch.Tensor)) else extras["episode"])
                    elif "log" in extras and extras["log"] is not None:
                        ep_infos.append(extras["log"][env_idx] if isinstance(extras["log"], (list, torch.Tensor)) else extras["log"])
                    
                    rewbuffer.append(cur_reward_sum[env_idx].item())
                    lenbuffer.append(cur_episode_length[env_idx].item())
                    
                    # Reset tracking
                    cur_reward_sum[env_idx] = 0
                    cur_episode_length[env_idx] = 0
                    expert_indices[env_idx] = 0
                    prev_latent[env_idx] = 0
                    prev_action[env_idx] = 0
            
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)
            
            global_step += 1
        
        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        fps = int(self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        if locs["loss_dict"]:
            for key, value in locs["loss_dict"].items():
                self.writer.add_scalar(f"Loss/{key}", value, locs["it"])

        # -- Training (SB3-style: always log, use completed episodes or current running mean)
        if len(locs["rewbuffer"]) > 0:
            mean_reward = statistics.mean(locs["rewbuffer"])
            mean_ep_len = statistics.mean(locs["lenbuffer"])
            self.writer.add_scalar("Train/mean_reward", mean_reward, locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", mean_ep_len, locs["it"])
            self.writer.add_scalar("Train/mean_reward/completed_episodes", mean_reward, locs["it"])
        
        # Always log current running reward (like SB3 rollout/ep_rew_mean)
        if locs["cur_reward_sum"].numel() > 0:
            running_mean_reward = locs["cur_reward_sum"].mean().item()
            self.writer.add_scalar("Train/running_reward", running_mean_reward, locs["it"])

        # callback for video logging
        if self.logger_type in ["wandb"]:
            self.writer.callback(locs["it"])

        str_msg = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str_msg.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {locs["learn_time"]:.3f}s)\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"Mean {key}:":>{pad}} {value:.4f}\n"""
            # -- Rewards
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
            # -- episode info
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""
        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str_msg.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {locs["learn_time"]:.3f}s)\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {
                time.strftime(
                    "%H:%M:%S",
                    time.gmtime(
                        self.tot_time
                        / (locs["it"] - locs["start_iter"] + 1)
                        * (locs["start_iter"] + locs["num_learning_iterations"] - locs["it"])
                    ),
                )
            }\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "actor_state_dict": self.alg.actor.state_dict(),
            "critic_state_dict": self.alg.critic.state_dict(),
            "target_critic_state_dict": self.alg.target_critic.state_dict(),
            "actor_optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.alg.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # -- Load model
        self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"])
        self.alg.critic.load_state_dict(loaded_dict["critic_state_dict"])
        self.alg.target_critic.load_state_dict(loaded_dict["target_critic_state_dict"])
        self.log_alpha = loaded_dict["log_alpha"]
        self.alpha = self.log_alpha.exp()
        # -- load optimizer if used
        if load_optimizer:
            self.alg.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.alg.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            self.alpha_optimizer.load_state_dict(loaded_dict["alpha_optimizer_state_dict"])
        # -- load current learning iteration
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode
        if device is not None:
            self.alg.actor.to(device)
        return lambda obs: self.alg.actor(obs)[0]  # Return action only

    def train_mode(self):
        if hasattr(self.alg, "actor"):
            self.alg.actor.train()
        if hasattr(self.alg, "critic"):
            self.alg.critic.train()

    def eval_mode(self):
        if hasattr(self.alg, "actor"):
            self.alg.actor.eval()
        if hasattr(self.alg, "critic"):
            self.alg.critic.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

    def _construct_algorithm(self, obs: torch.Tensor):
        """Construct the SAC algorithm."""
        # Import SAC modules
        from robot_rl.modules.sac import Actor, Critic
        import copy
        
        # Get observation and action dimensions dynamically
        base_obs_dim = obs.shape[-1]
        action_dim = self.env.num_actions
        state_dim = base_obs_dim * 2 + action_dim
        
        # Create actor and critic networks
        actor = Actor(
            obs_dim=state_dim,
            num_actions=action_dim,
            hidden_dims=self.policy_cfg.get("actor_hidden_dims", [256, 256, 256]),
            activation=self.policy_cfg.get("activation", "elu"),
            max_action=self.policy_cfg.get("max_action", 1.0),
        ).to(self.device)
        
        critic = Critic(
            obs_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.policy_cfg.get("critic_hidden_dims", [256, 256, 256]),
            activation=self.policy_cfg.get("activation", "elu"),
        ).to(self.device)
        
        # Create target critic (for stable Q-learning)
        target_critic = copy.deepcopy(critic)
        target_critic.eval()  # Target network is not trained directly
        
        # Create optimizers
        lr = self.alg_cfg.get("learning_rate", 3e-4)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
        
        # Store in a simple container
        class SACAlgorithm:
            def __init__(self, actor, critic, target_critic, actor_opt, critic_opt):
                self.actor = actor
                self.critic = critic
                self.target_critic = target_critic
                self.actor_optimizer = actor_opt
                self.critic_optimizer = critic_opt
        
        return SACAlgorithm(actor, critic, target_critic, actor_optimizer, critic_optimizer)

    def _prepare_logging_writer(self) -> None:
        """Prepares the logging writers."""
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from robot_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from robot_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")
    
    def _sac_update(self, replay_buffer, buffer_size):
        """Perform SAC gradient update with multiple gradient steps."""
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_alpha_loss = 0.0
        total_q = 0.0
        
        # Perform multiple gradient steps (batch collection pattern)
        for _ in range(self.gradient_steps):
            # Sample random batch efficiently from numpy buffer
            indices = np.random.randint(0, buffer_size, size=self.batch_size)
            
            states = torch.as_tensor(replay_buffer["states"][indices], device=self.device)
            actions = torch.as_tensor(replay_buffer["actions"][indices], device=self.device)
            rewards = torch.as_tensor(replay_buffer["rewards"][indices], device=self.device)
            next_states = torch.as_tensor(replay_buffer["next_states"][indices], device=self.device)
            dones = torch.as_tensor(replay_buffer["dones"][indices], device=self.device)
            
            # Update Critic (Q-functions)
            with torch.no_grad():
                # Sample next actions from current policy
                next_actions, next_log_probs = self.alg.actor(next_states)
                
                # Compute target Q-values using target critic
                target_q1, target_q2 = self.alg.target_critic(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2)
                
                # Add entropy regularization to target: Q_target = r + γ(Q(s',a') - α*log π(a'|s'))
                target_q = target_q - self.alpha.detach() * next_log_probs
                
                # Compute TD target: y = r + γ * (1 - done) * Q_target
                td_target = rewards + self.gamma * (1 - dones.float()) * target_q
            
            # Compute current Q-values
            current_q1, current_q2 = self.alg.critic(states, actions)
            
            # Compute critic loss (MSE between current Q and TD target)
            critic_loss = torch.nn.functional.mse_loss(current_q1, td_target) + \
                          torch.nn.functional.mse_loss(current_q2, td_target)
            
            # Update critic
            self.alg.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.alg.critic_optimizer.step()
            
            # Update Actor (Policy)
            # Sample actions from current policy
            new_actions, log_probs = self.alg.actor(states)
            
            # Compute Q-values for new actions
            q1_new, q2_new = self.alg.critic(states, new_actions)
            q_new = torch.min(q1_new, q2_new)
            
            # Actor loss: maximize Q-value with entropy regularization
            # J_π = E[α*log π(a|s) - Q(s,a)]
            actor_loss = (self.alpha.detach() * log_probs - q_new).mean()
            
            # Update actor
            self.alg.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.alg.actor_optimizer.step()
            
            # Update Temperature (Alpha)
            # Automatic entropy tuning: adjust alpha to match target entropy
            # J_α = E[-α * (log π(a|s) + H_target)]
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Update alpha value
            self.alpha = self.log_alpha.exp()
            
            # Update Target Networks (Soft Update)
            # θ_target = τ * θ + (1 - τ) * θ_target
            with torch.no_grad():
                for target_param, param in zip(self.alg.target_critic.parameters(), self.alg.critic.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # Accumulate losses for logging
            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
            total_alpha_loss += alpha_loss.item()
            total_q += q_new.mean().item()
        
        # Average losses over gradient steps
        loss_dict = {
            "critic_loss": total_critic_loss / self.gradient_steps,
            "actor_loss": total_actor_loss / self.gradient_steps,
            "alpha_loss": total_alpha_loss / self.gradient_steps,
            "alpha": self.alpha.item(),
            "mean_q": total_q / self.gradient_steps,
        }
        
        return loss_dict