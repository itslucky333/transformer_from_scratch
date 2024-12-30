import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # Learnable scaling factor
        self.bias = nn.Parameter(torch.zeros(features))  # Learnable bias term

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # Compute mean along the last dimension
        std = x.std(dim=-1, keepdim=True)    # Compute standard deviation along the last dimension
        return self.alpha * (x - mean) / (std + self.eps) + self.bias  # Normalize and scale



# Define LayerNormalization
layer_norm = LayerNormalization(features=10)

# Input tensor (batch_size=2, seq_len=5, features=10)
input_tensor = torch.randn(2, 5, 10)

# Apply layer normalization
output = layer_norm(input_tensor)

# Output shape
print("Output shape:", output.shape)  # Expected: [2, 5, 10]
