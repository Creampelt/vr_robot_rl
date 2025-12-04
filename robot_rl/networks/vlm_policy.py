# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision-Language conditioned policy using camera_latent.

This module conditions a small language model via soft prompts generated from
`camera_latent` to produce concise navigation text, which is then embedded
into fixed-size vectors for ActorCritic. The structure ensures compatibility
with PPO storage and evaluation.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


class SoftPromptTextEncoder(nn.Module):
    """Condition a small LM with camera_latent via soft prompts to generate text."""

    def __init__(self, lm_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", soft_prompt_len: int = 16, latent_dim: int = 384, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name, torch_dtype=torch.float16).to(device)
        self.lm.eval()
        hidden_size = self.lm.config.hidden_size
        # Map incoming latents to soft_prompt_len * hidden_size; allow lazy in_features
        try:
            self.soft_prompt = nn.LazyLinear(soft_prompt_len * hidden_size)
        except Exception:
            # Fallback to fixed Linear if LazyLinear unavailable
            self.soft_prompt = nn.Linear(latent_dim, soft_prompt_len * hidden_size)
        # Ensure soft prompt projection matches LM dtype (fp16)
        self.soft_prompt = self.soft_prompt.to(device).half()
        self.soft_prompt_len = soft_prompt_len

    def forward(self, latents: torch.Tensor, instruction: str) -> List[str]:
        B = latents.shape[0]
        with torch.no_grad():
            # Ensure dtype/device compatibility for Linear
            # Ensure dtype/device compatibility for Linear
            # Force latents to fp16 for LM compatibility
            latents_fp = latents.to(dtype=torch.float16, device=self.device)
            prompt_embeds = self.soft_prompt(latents_fp).view(B, self.soft_prompt_len, self.lm.config.hidden_size)
            toks = self.tokenizer([instruction] * B, return_tensors="pt", padding=True).to(self.device)
            input_ids = toks.input_ids
            base_embeds = self.lm.get_input_embeddings()(input_ids).to(dtype=torch.float16)
            inputs_embeds = torch.cat([prompt_embeds, base_embeds], dim=1)
            generated = self.lm.generate(inputs_embeds=inputs_embeds, max_new_tokens=32, do_sample=False)
            texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return texts

class TextEmbedder(nn.Module):
    """Embeds text descriptions into fixed-size vectors using sentence transformers."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        embedding_dim: int = 768,
        device: str = "cuda"
    ):
        """Initialize the text embedder.
        
        Recommended models (ordered by quality):
        - "sentence-transformers/all-mpnet-base-v2" (768-dim, 420MB) - Default, best quality
        - "sentence-transformers/all-MiniLM-L12-v2" (384-dim, 120MB) - Great balance
        - "sentence-transformers/all-MiniLM-L6-v2" (384-dim, 80MB) - Fastest, still good
        - "BAAI/bge-small-en-v1.5" (384-dim, 133MB) - Strong retrieval performance
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
            embedding_dim: Dimension of output embeddings
            device: Device to run the model on
        """
        super().__init__()
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device
        
        print(f"[TextEmbedder] Loading text embedding model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Embed text descriptions into fixed-size vectors.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Tensor of embeddings, shape (B, embedding_dim)
        """
        with torch.no_grad():
            # Tokenize texts
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            
            # Get model output
            model_output = self.model(**encoded_input)
            
            # Perform mean pooling
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class VLMObservationEncoder(nn.Module):
    """Processes camera_latent via soft-prompt LM -> text -> embeddings."""
    def __init__(
        self,
        lm_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        text_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        embedding_dim: int = 768,
        latent_dim: int = 384,
        prompt_template: str = "You are a navigation expert. Based on the latent features, describe the immediate action (direction and speed) to move safely towards the goal.",
        soft_prompt_len: int = 16,
        device: str = "cuda"
    ):
        """Initialize the VLM observation encoder.
        
        Args:
            vlm_model_name: HuggingFace model name for the VLM
            text_model_name: HuggingFace model name for text embeddings
            embedding_dim: Dimension of text embeddings
            prompt_template: Prompt template for VLM (None uses default temporal prompt)
            max_new_tokens: Max tokens to generate
            use_temporal_context: Whether to use world model rollouts if provided
            device: Device to run on
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.device = device
        self.prompt_template = prompt_template
        # Use latent_dim for camera_latent input width
        self.vlm_encoder = SoftPromptTextEncoder(lm_name=lm_name, soft_prompt_len=soft_prompt_len, latent_dim=latent_dim, device=device)
        
        # Initialize text embedder
        self.text_embedder = TextEmbedder(
            model_name=text_model_name,
            embedding_dim=embedding_dim,
            device=device
        )
        
        # Freeze both VLM and text embedder (don't train them)
        for param in self.vlm_encoder.parameters():
            param.requires_grad = False
        for param in self.text_embedder.parameters():
            param.requires_grad = False
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        texts = self.vlm_encoder(latents, self.prompt_template)
        embeddings = self.text_embedder(texts)
        return embeddings
    
    def get_text_descriptions(self, images: torch.Tensor, world_model_rollout: torch.Tensor = None) -> List[str]:
        """Get the actual text descriptions (useful for debugging/visualization).
        
        Args:
            images: Batched RGB images
            world_model_rollout: Optional predicted future frames
            
        Returns:
            List of text descriptions
        """
        return self.vlm_encoder(images, world_model_rollout)

class VLMActorCritic(nn.Module):
    """Actor-Critic network that uses VLM-generated text embeddings as observations.
    
    This wraps the VLMObservationEncoder with robot_rl's standard ActorCritic module.
    The architecture is:
    
    Camera Images -> VLM -> Text -> Text Embeddings -> ActorCritic (from robot_rl.modules)
    
    Compatible with PPO training from robot_rl/rsl_rl.
    """
    
    def __init__(
        self,
        vlm_encoder: VLMObservationEncoder,
        num_actions: int,
        num_critic_obs: int = None,
        actor_hidden_dims: List[int] = [256, 128, 64],
        critic_hidden_dims: List[int] = [256, 128, 64],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        noise_std_type: str = "scalar"
    ):
        """Initialize the VLM Actor-Critic.
        
        Args:
            vlm_encoder: The VLM observation encoder (frozen)
            num_actions: Number of action dimensions
            num_critic_obs: Total dimension of critic observations (VLM embeddings + velocity + actions + clock, etc.).
                          If None, defaults to vlm_encoder.embedding_dim (symmetric actor-critic)
            actor_hidden_dims: Hidden layer dimensions for actor MLP
            critic_hidden_dims: Hidden layer dimensions for critic MLP
            activation: Activation function
            init_noise_std: Initial standard deviation for action noise
            actor_obs_normalization: Whether to normalize actor observations
            critic_obs_normalization: Whether to normalize critic observations
            noise_std_type: Type of noise std ('scalar' or 'log')
        """
        super().__init__()
        
        self.vlm_encoder = vlm_encoder
        self.num_actions = num_actions
        # Mirror encoder's prompt template for convenience
        self.prompt_template = getattr(vlm_encoder, "prompt_template", "")
        self.actor_obs_dim = vlm_encoder.embedding_dim
        
        # Import ActorCritic from robot_rl.modules
        from robot_rl.modules import ActorCritic
        from tensordict import TensorDict
        
        # Create fake observation TensorDict for ActorCritic initialization
        # ActorCritic expects obs groups, we'll use "policy" and "critic"
        actor_obs_dim = vlm_encoder.embedding_dim
        
        # If num_critic_obs not provided, use symmetric actor-critic (same as actor)
        if num_critic_obs is None:
            num_critic_obs = actor_obs_dim
            print(f"[VLMActorCritic] Using symmetric actor-critic with {actor_obs_dim}-dim observations")
        else:
            print(f"[VLMActorCritic] Using asymmetric actor-critic: actor={actor_obs_dim}-dim, critic={num_critic_obs}-dim")
        self.critic_obs_dim = num_critic_obs
        
        fake_obs = TensorDict({
            "policy": torch.zeros(1, actor_obs_dim),
            "critic": torch.zeros(1, num_critic_obs)
        }, batch_size=[1])
        
        obs_groups = {
            "policy": ["policy"],
            "critic": ["critic"]
        }
        
        # Create ActorCritic using robot_rl's standard module
        self.actor_critic = ActorCritic(
            obs=fake_obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type
        )
        
        print(f"[VLMActorCritic] Created with {actor_obs_dim}-dim actor obs, {num_critic_obs}-dim critic obs -> {num_actions} actions")
    
    def forward(self, observations: torch.Tensor):
        """Forward pass for actor-critic.
        
        Args:
            observations: Camera images (BxHxWx3) or pre-computed text embeddings (Bxembedding_dim)
            
        Returns:
            TensorDict with actor and critic outputs
        """
        # Observations expected to be camera_latent (BxD)
        embeddings = observations if observations.ndim == 2 else observations
        
        # Create TensorDict with embeddings for both policy and critic
        from tensordict import TensorDict
        obs_td = TensorDict({
            "policy": embeddings,
            "critic": embeddings
        }, batch_size=[embeddings.shape[0]])
        
        # Forward through ActorCritic
        return self.actor_critic.act_inference(obs_td)
    
    def act(self, observations, **kwargs):
        """Forward wrapper for compatibility with PPO training."""
        from tensordict import TensorDict
        
        # Check if observations are already processed embeddings (flat tensors)
        if isinstance(observations, TensorDict) and "policy" in observations and "critic" in observations:
            policy_obs = observations["policy"]
            critic_obs = observations["critic"]
            
            # Both should be flat 2D tensors if already processed
            policy_processed = isinstance(policy_obs, torch.Tensor) and policy_obs.ndim == 2 and policy_obs.shape[1] in [384, 768, 1024]
            critic_processed = isinstance(critic_obs, torch.Tensor) and critic_obs.ndim == 2
            
            if policy_processed and critic_processed:
                # Already processed - pass directly
                return self.actor_critic.act(observations, **kwargs)
        
        # Process observations through VLM encoder to get embeddings
        obs_td = self._process_observations(observations)
        return self.actor_critic.act(obs_td, **kwargs)
    
    def act_inference(self, observations, **kwargs):
        """Inference wrapper for compatibility."""
        from tensordict import TensorDict
        
        # Check if observations are already processed embeddings (flat tensors)
        if isinstance(observations, TensorDict) and "policy" in observations and "critic" in observations:
            policy_obs = observations["policy"]
            critic_obs = observations["critic"]
            
            policy_processed = isinstance(policy_obs, torch.Tensor) and policy_obs.ndim == 2 and policy_obs.shape[1] in [384, 768, 1024]
            critic_processed = isinstance(critic_obs, torch.Tensor) and critic_obs.ndim == 2
            
            if policy_processed and critic_processed:
                # Already processed
                return self.actor_critic.act_inference(observations, **kwargs)
        
        obs_td = self._process_observations(observations)
        return self.actor_critic.act_inference(obs_td, **kwargs)
    
    def evaluate(self, observations, **kwargs):
        """Evaluate wrapper for compatibility."""
        from tensordict import TensorDict
        
        # During PPO updates, observations from buffer are already processed
        # Check if they're already in the right format (flat tensors)
        if isinstance(observations, TensorDict) and "policy" in observations and "critic" in observations:
            policy_obs = observations["policy"]
            critic_obs = observations["critic"]
            
            policy_is_processed = (isinstance(policy_obs, torch.Tensor) and 
                                  policy_obs.ndim == 2 and 
                                  policy_obs.shape[1] in [384, 768, 1024])
            critic_is_processed = isinstance(critic_obs, torch.Tensor) and critic_obs.ndim == 2
            
            if policy_is_processed and critic_is_processed:
                # Already processed, pass directly
                return self.actor_critic.evaluate(observations, **kwargs)
        
        # Not yet processed, process observations through VLM
        obs_td = self._process_observations(observations)
        return self.actor_critic.evaluate(obs_td, **kwargs)
    
    def _process_observations(self, observations):
        """Process observations through VLM encoder and create TensorDict.
        
        Args:
            observations: TensorDict with camera images or embeddings
            
        Returns:
            TensorDict with VLM embeddings for policy and critic
        """
        from tensordict import TensorDict

        # Helpers: ensure tensors are (B, D) and align dtype/device
        def _flatten_2d(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 2:
                return x
            if x.ndim == 1:
                return x.unsqueeze(-1)
            if x.ndim == 0:
                return x.view(1, 1)
            return x.flatten(start_dim=1)

        def _align_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            return x.to(dtype=ref.dtype, device=ref.device)

        def _match_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
            x = _flatten_2d(x)
            cur = x.shape[1]
            if cur == dim:
                return x
            if cur > dim:
                return x[:, :dim]
            # pad zeros to the right
            pad = torch.zeros(x.shape[0], dim - cur, dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=1)

        # 1) Extract camera_latent for policy
        latents: torch.Tensor
        critic_obs = None

        if isinstance(observations, TensorDict):
            # Prefer nested groups
            if "policy" in observations:
                policy_group = observations["policy"]
                if isinstance(policy_group, TensorDict):
                    # camera_latent under policy group
                    if "camera_latent" in policy_group:
                        latents = _flatten_2d(policy_group["camera_latent"])
                    # Sometimes camera key reused for latents
                    elif "camera" in policy_group:
                        latents = _flatten_2d(policy_group["camera"])
                    else:
                        # If policy is already a tensor (embeddings/latents)
                        raise ValueError(f"Policy group missing 'camera_latent'/'camera'. Keys: {list(policy_group.keys())}")
                elif isinstance(policy_group, torch.Tensor):
                    latents = _flatten_2d(policy_group)
                else:
                    raise ValueError("Unsupported policy group type")
            elif "camera_latent" in observations:
                latents = _flatten_2d(observations["camera_latent"])
            elif "camera" in observations:
                latents = _flatten_2d(observations["camera"])  # expected to be latents
            else:
                raise ValueError(f"Could not find camera_latent. Top-level keys: {list(observations.keys())}")

            # Critic obs (optional, for asymmetric)
            critic_obs = observations.get("critic", None)
        else:
            # Observations directly are latents/embeddings
            latents = _flatten_2d(observations)

        if latents.ndim != 2:
            raise ValueError(f"Expected 2D latent tensor, got shape: {latents.shape}")

        # 2) Encode latents via soft-prompt LM -> text -> embeddings
        # Encode latents via VLMObservationEncoder (uses internal prompt)
        embeddings = self.vlm_encoder(latents)

        # 3) Build critic vector
        if critic_obs is not None:
            if isinstance(critic_obs, TensorDict):
                parts = []
                # If critic has its own camera_latent, encode similarly
                if "camera_latent" in critic_obs:
                    c_lat = _flatten_2d(critic_obs["camera_latent"])
                    c_emb = self.vlm_encoder(_align_like(c_lat, embeddings))
                    parts.append(c_emb)
                elif "camera" in critic_obs:
                    c_lat = _flatten_2d(critic_obs["camera"])  # expected to be latents
                    c_emb = self.vlm_encoder(_align_like(c_lat, embeddings))
                    parts.append(c_emb)
                else:
                    # No camera component; actor embeddings may still be enough
                    parts.append(embeddings)

                # Append other critic features
                for key in sorted(critic_obs.keys()):
                    if key in ("camera", "camera_latent"):
                        continue
                    val = critic_obs[key]
                    if isinstance(val, torch.Tensor):
                        parts.append(_align_like(_flatten_2d(val), embeddings))

                critic_vec = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
            elif isinstance(critic_obs, torch.Tensor):
                # Ensure critic always includes VLM embedding + extras
                critic_extra = _align_like(_flatten_2d(critic_obs), embeddings)
                critic_vec = torch.cat([embeddings, critic_extra], dim=-1)
            else:
                critic_vec = embeddings
        else:
            critic_vec = embeddings

        # 4) Ensure dims match ActorCritic expectations
        embeddings = _match_dim(embeddings, self.actor_obs_dim)
        critic_vec = _match_dim(critic_vec, self.critic_obs_dim)

        # 5) Return standardized TensorDict for ActorCritic
        from tensordict import TensorDict as TD
        return TD({
            "policy": embeddings,
            "critic": critic_vec
        }, batch_size=[embeddings.shape[0]])
    
    def reset(self, dones=None):
        """Reset wrapper for compatibility."""
        return self.actor_critic.reset(dones)
    
    def get_text_descriptions(self, observations: torch.Tensor) -> List[str]:
        if observations.ndim == 2:
            # Return texts generated by the inner soft-prompt LM
            return self.vlm_encoder.vlm_encoder(observations, getattr(self.vlm_encoder, "prompt_template", ""))
        else:
            return ["(Expected camera_latent)"] * observations.shape[0]
    
    @property
    def is_recurrent(self):
        return self.actor_critic.is_recurrent
    
    @property
    def action_std(self):
        """Delegate to wrapped actor_critic."""
        return self.actor_critic.action_std
    
    @property
    def action_mean(self):
        """Delegate to wrapped actor_critic."""
        return self.actor_critic.action_mean

    @property
    def entropy(self):
        """Expose latest policy entropy recorded by inner ActorCritic."""
        return getattr(self.actor_critic, "entropy", None)
    
    def get_actions_log_prob(self, actions):
        """Get log probabilities for given actions - required by PPO.
        
        Note: This method doesn't receive observations as parameter.
        PPO/algorithms access observations from their own storage buffer.
        We delegate directly to the wrapped actor_critic.
        """
        return self.actor_critic.get_actions_log_prob(actions)
    
    def get_values(self, observations):
        """Get value estimates - required by PPO."""
        obs_td = self._process_observations(observations)
        return self.actor_critic.get_values(obs_td)
    
    def update(self):
        """Update normalizers if present."""
        if hasattr(self.actor_critic, 'update'):
            return self.actor_critic.update()
    
    def update_normalization(self, obs, **kwargs):
        """Update observation normalization - required by PPO."""
        if hasattr(self.actor_critic, 'update_normalization'):
            # Process observations through VLM if needed
            obs_td = self._process_observations(obs)
            # Process last_obs if provided
            if 'last_obs' in kwargs and kwargs['last_obs'] is not None:
                kwargs['last_obs'] = self._process_observations(kwargs['last_obs'])
            return self.actor_critic.update_normalization(obs_td, **kwargs)
    
    def train(self, mode=True):
        """Set training mode."""
        super().train(mode)
        self.actor_critic.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
    
    def state_dict(self, *args, **kwargs):
        """Get state dict - delegate to actor_critic."""
        return self.actor_critic.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load state dict - delegate to actor_critic."""
        return self.actor_critic.load_state_dict(state_dict, *args, **kwargs)
    
    def to(self, device):
        """Move to device."""
        super().to(device)
        self.vlm_encoder.to(device)
        self.actor_critic.to(device)
        return self


# Convenience function to create a VLM-based actor-critic
def create_vlm_actor_critic(
    num_actions: int,
    num_critic_obs: int = None,
    lm_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    text_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    soft_prompt_len: int = 16,
    latent_dim: int = 384,
    actor_hidden_dims: List[int] = [256, 128, 64],
    critic_hidden_dims: List[int] = [256, 128, 64],
    activation: str = "elu",
    init_noise_std: float = 1.0,
    actor_obs_normalization: bool = False,
    critic_obs_normalization: bool = False,
    noise_std_type: str = "scalar",
    device: str = "cuda"
) -> VLMActorCritic:
    """Factory function to create a VLM-based actor-critic network.
    
    Default configuration uses:
    - Qwen2-VL-2B: Best quality-to-size ratio VLM (2B params)
    - all-mpnet-base-v2: Best quality sentence embeddings (768-dim)
    
    Args:
        num_actions: Number of action dimensions
        num_critic_obs: Total dimension of critic observations (embeddings + velocity + actions + clock, etc.).
                       If None, uses symmetric actor-critic (same as actor)
        vlm_model_name: VLM model name (default: Qwen2-VL-2B, SOTA for 2B params)
        text_model_name: Text embedding model name (default: all-mpnet-base-v2, best quality)
        actor_hidden_dims: Actor MLP hidden dimensions
        critic_hidden_dims: Critic MLP hidden dimensions
        activation: Activation function
        init_noise_std: Initial action noise std
        actor_obs_normalization: Normalize actor observations
        critic_obs_normalization: Normalize critic observations
        noise_std_type: Type of noise std
        device: Device to run on
        
    Returns:
        VLMActorCritic network ready for PPO training
    """
    # Create VLM observation encoder
    vlm_encoder = VLMObservationEncoder(
        lm_name=lm_name,
        text_model_name=text_model_name,
        soft_prompt_len=soft_prompt_len,
        latent_dim=latent_dim,
        device=device
    )
    
    # Create actor-critic
    actor_critic = VLMActorCritic(
        vlm_encoder=vlm_encoder,
        num_actions=num_actions,
        num_critic_obs=num_critic_obs,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        activation=activation,
        init_noise_std=init_noise_std,
        actor_obs_normalization=actor_obs_normalization,
        critic_obs_normalization=critic_obs_normalization,
        noise_std_type=noise_std_type
    )
    
    return actor_critic