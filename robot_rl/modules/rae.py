import torch
import torch.nn as nn
from transformers import TorchAoConfig, AutoImageProcessor, AutoModel
from torchao.quantization import Int4WeightOnlyConfig

class RAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitsplus-pretrain-lvd1689m")
        # quantization to reduce memory usage
        quant_type = Int4WeightOnlyConfig(group_size=128)
        quantization_config = TorchAoConfig(quant_type=quant_type)

        self.model = AutoModel.from_pretrained(
            "facebook/dinov3-vit7b16-pretrain-lvd1689m",
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config
            # add_pooling_layer=True
        )

    def forward(self, image): # recommended shape - (3, 224, 224), the image should be a PIL image
        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        # Use CLS token (first token) as single feature vector
        pooled_output = outputs.pooler_output
        return pooled_output
