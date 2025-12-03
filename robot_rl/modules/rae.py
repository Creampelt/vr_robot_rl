import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class RAE(nn.Module):
    def __init__(self, pretrained_model_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m") -> None:
        super().__init__()
        self.feature_dim = 384
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name, use_fast=True)

        self.model = AutoModel.from_pretrained(
            pretrained_model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode batched images into feature vectors.

        Args:
            images (torch.Tensor): Batched RGB images. Shape is Bx3xHxW, where H and W are divisible by 16 and >= 224.

        Returns:
            test (torch.Tensor): Feature vector of images. Shape is Bx384.
        """
        _, H, W, C = images.shape
        assert (H % 16 == 0) and (W % 16 == 0) and H > 224 and W > 224 and C == 3, (
            f"Height, width, and/or channels of image do not meet requirements. Got {H=}, {W=}, {C=}"
        )
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        # Use CLS token (first token) as single feature vector
        pooled_output = outputs.pooler_output
        return pooled_output
