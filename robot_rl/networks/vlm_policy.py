# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision-Language Model (VLM) based policy for embodied navigation.

This module implements a VLM-based policy where:
1. Camera observations are processed by a VLM to generate text descriptions
2. Text descriptions are embedded using a sentence transformer  
3. Text embeddings are used by robot_rl's ActorCritic module for action prediction
4. Compatible with PPO training
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from typing import List, Dict
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer, AutoModel


class VLMTextEncoder(nn.Module):
    """Efficient Vision-Language Model that converts camera images to text descriptions.
    
    Supports both single-frame and multi-frame (world model rollout) processing.
    Uses small, efficient VLMs optimized for edge deployment and fast inference.
    """
    
    def __init__(
        self,
        vlm_model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        prompt_template: str = "You are a navigation expert helping a mobile robot. Based on the sequence of predicted future observations from the world model, describe the optimal navigation action (direction and speed) the robot should take to safely reach its goal efficiently. Consider obstacles, free space, and goal direction.",
        max_new_tokens: int = 50,
        use_temporal_context: bool = True,
        device: str = "cuda"
    ):
        """Initialize the VLM text encoder.
        
        Recommended small VLMs (ordered by quality):
        - "Qwen/Qwen2-VL-2B-Instruct" (2B params) - Default, SOTA quality for size, very fast
        - "microsoft/Phi-3.5-vision-instruct" (4.2B params) - Excellent reasoning
        - "vikhyatk/moondream2" (1.8B params) - Fastest, good for edge
        - "OpenGVLab/InternVL2-2B" (2B params) - Strong vision understanding
        
        Args:
            vlm_model_name: HuggingFace model name for the VLM
            prompt_template: Text prompt for better descriptions (designed for temporal reasoning)
            max_new_tokens: Maximum tokens to generate (keep moderate for temporal reasoning)
            use_temporal_context: If True, processes multiple frames as temporal sequence
            device: Device to run the model on
        """
        super().__init__()
        
        self.vlm_model_name = vlm_model_name
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.use_temporal_context = use_temporal_context
        self.device = device
        
        # Get HuggingFace token if needed
        hf_token = os.environ.get("HF_TOKEN", None)
        
        print(f"[VLMTextEncoder] Loading VLM model: {vlm_model_name}")
        print(f"[VLMTextEncoder] Temporal context: {use_temporal_context}")
        
        # Load VLM processor and model
        self.processor = AutoProcessor.from_pretrained(
            vlm_model_name,
            token=hf_token,
            trust_remote_code=True
        )
        
        self.vlm_model = AutoModelForVision2Seq.from_pretrained(
            vlm_model_name,
            torch_dtype=torch.float16,  # Use FP16 for efficiency
            token=hf_token,
            trust_remote_code=True
        ).to(device)
        
        self.vlm_model.eval()
        
    def forward(self, images: torch.Tensor, world_model_rollout: torch.Tensor = None) -> List[str]:
        """Generate text descriptions from camera images with optional world model rollout.
        
        Args:
            images: Current observation images. Shape is BxHxWx3 or Bx3xHxW
            world_model_rollout: Optional predicted future frames from world model.
                                Shape is BxTxHxWx3 or BxTx3xHxW where T is rollout horizon
            
        Returns:
            List of text descriptions with action recommendations
        """
        B = images.shape[0]
        
        # Use world model rollout if provided and temporal context is enabled
        if self.use_temporal_context and world_model_rollout is not None:
            # World model rollout: BxTxHxWx3 or BxTx3xHxW
            if world_model_rollout.ndim == 5:
                T = world_model_rollout.shape[1]
                
                # Convert from BxTxHxWxC to BxTxCxHxW if needed
                if world_model_rollout.shape[-1] == 3:
                    world_model_rollout = world_model_rollout.permute(0, 1, 4, 2, 3)
                
                # Convert to float [0, 1] if needed
                if world_model_rollout.dtype == torch.uint8:
                    world_model_rollout = world_model_rollout.float() / 255.0
                
                # Create enhanced prompt with temporal information
                enhanced_prompts = [
                    f"{self.prompt_template} Here are {T} predicted future frames showing what the robot will see if it continues. Frame 1 is immediate future, Frame {T} is furthest. What action should the robot take NOW?"
                ] * B
                
                # Process multiple frames per batch item
                # Flatten to (B*T)x3xHxW for processing
                all_frames = world_model_rollout.reshape(B * T, *world_model_rollout.shape[2:])
                
                with torch.no_grad():
                    # Note: Some VLMs support multi-image input natively
                    # For simplicity, we'll process frames sequentially and let the VLM reason about them
                    inputs = self.processor(
                        images=list(all_frames),
                        text=enhanced_prompts * T,  # Repeat prompt for each frame
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    generated_ids = self.vlm_model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        num_beams=1
                    )
                    
                    generated_texts = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )
                    
                    # Take first B descriptions (one per batch item)
                    generated_texts = generated_texts[:B]
            else:
                # Fallback to single frame if rollout shape is unexpected
                generated_texts = self._process_single_frame(images, B)
        else:
            # Process single frame (current observation only)
            generated_texts = self._process_single_frame(images, B)
        
        return generated_texts
    
    def _process_single_frame(self, images: torch.Tensor, B: int) -> List[str]:
        """Process single frame observations."""
        # Convert from BxHxWxC to BxCxHxW if needed
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        
        # Convert to float [0, 1] if needed
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        
        # Simple prompt for single frame
        single_frame_prompt = "You are a navigation expert. Based on this camera view, what navigation action (direction and speed) should the robot take to reach its goal safely and efficiently?"
        
        with torch.no_grad():
            prompts = [single_frame_prompt] * B
            inputs = self.processor(
                images=list(images),
                text=prompts,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            generated_ids = self.vlm_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1
            )
            
            generated_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
        
        return generated_texts


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
    """Processes camera observations through VLM+text embeddings.
    
    Supports world model rollouts for improved temporal reasoning.
    This module combines VLMTextEncoder and TextEmbedder to convert
    camera images (and optional predicted future frames) into fixed-size 
    embeddings that can be used as observations by standard actor-critic networks.
    """
    def __init__(
        self,
        vlm_model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        text_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        embedding_dim: int = 768,
        prompt_template: str = None,  # Will use VLMTextEncoder default
        max_new_tokens: int = 50,
        use_temporal_context: bool = True,
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
        self.use_temporal_context = use_temporal_context
        
        # Initialize VLM text encoder with temporal reasoning support
        vlm_kwargs = {
            "vlm_model_name": vlm_model_name,
            "max_new_tokens": max_new_tokens,
            "use_temporal_context": use_temporal_context,
            "device": device
        }
        if prompt_template is not None:
            vlm_kwargs["prompt_template"] = prompt_template
            
        self.vlm_encoder = VLMTextEncoder(**vlm_kwargs)
        
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
    
    def forward(self, images: torch.Tensor, world_model_rollout: torch.Tensor = None) -> torch.Tensor:
        """Process images (and optional world model rollout) through VLM to get text embeddings.
        
        Args:
            images: Current observation images. Shape is BxHxWx3 or Bx3xHxW
            world_model_rollout: Optional predicted future frames. Shape is BxTxHxWx3 or BxTx3xHxW
            
        Returns:
            Text embeddings, shape (B, embedding_dim)
        """
        # Generate text descriptions from images with world model context
        texts = self.vlm_encoder(images, world_model_rollout)
        
        # Embed texts into fixed-size vectors
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
        
        # Import ActorCritic from robot_rl.modules
        from robot_rl.modules import ActorCritic
        from tensordict import TensorDict
        
        # Create fake observation TensorDict for ActorCritic initialization
        # ActorCritic expects obs groups, we'll use "policy" and "critic" with VLM embeddings
        obs_dim = vlm_encoder.embedding_dim
        fake_obs = TensorDict({
            "policy": torch.zeros(1, obs_dim),
            "critic": torch.zeros(1, obs_dim)
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
        
        print(f"[VLMActorCritic] Created with {obs_dim}-dim VLM embeddings -> {num_actions} actions")
    
    def forward(self, observations: torch.Tensor):
        """Forward pass for actor-critic.
        
        Args:
            observations: Camera images (BxHxWx3) or pre-computed text embeddings (Bxembedding_dim)
            
        Returns:
            TensorDict with actor and critic outputs
        """
        # Check if observations are images or embeddings
        if observations.ndim == 4:  # Images: BxHxWx3
            # Process through VLM to get text embeddings
            embeddings = self.vlm_encoder(observations)
        else:  # Already embeddings: Bxembedding_dim
            embeddings = observations
        
        # Create TensorDict with embeddings for both policy and critic
        from tensordict import TensorDict
        obs_td = TensorDict({
            "policy": embeddings,
            "critic": embeddings
        }, batch_size=[embeddings.shape[0]])
        
        # Forward through ActorCritic
        return self.actor_critic.act_inference(obs_td)
    
    def act(self, observations, **kwargs):
        """Forward wrapper for compatibility."""
        return self.actor_critic.act(observations, **kwargs)
    
    def act_inference(self, observations, **kwargs):
        """Inference wrapper for compatibility."""
        return self.actor_critic.act_inference(observations, **kwargs)
    
    def evaluate(self, observations, **kwargs):
        """Evaluate wrapper for compatibility."""
        return self.actor_critic.evaluate(observations, **kwargs)
    
    def reset(self, dones=None):
        """Reset wrapper for compatibility."""
        return self.actor_critic.reset(dones)
    
    def get_text_descriptions(self, observations: torch.Tensor) -> List[str]:
        """Get text descriptions for visualization/debugging."""
        if observations.ndim == 4:  # Images
            return self.vlm_encoder.get_text_descriptions(observations)
        else:
            return ["(Pre-computed embeddings - no text available)"] * observations.shape[0]
    
    @property
    def is_recurrent(self):
        return self.actor_critic.is_recurrent


# Convenience function to create a VLM-based actor-critic
def create_vlm_actor_critic(
    num_actions: int,
    vlm_model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    text_model_name: str = "sentence-transformers/all-mpnet-base-v2",
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
        vlm_model_name=vlm_model_name,
        text_model_name=text_model_name,
        device=device
    )
    
    # Create actor-critic
    actor_critic = VLMActorCritic(
        vlm_encoder=vlm_encoder,
        num_actions=num_actions,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        activation=activation,
        init_noise_std=init_noise_std,
        actor_obs_normalization=actor_obs_normalization,
        critic_obs_normalization=critic_obs_normalization,
        noise_std_type=noise_std_type
    )
    
    return actor_critic
