import torch
import torch.nn as nn

class LinearRegressionPyTorch(nn.Module):
    """PyTorch linear regression model converted from scikit-learn."""
    def __init__(self, input_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
