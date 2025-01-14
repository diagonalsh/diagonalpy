import torch
import torch.nn as nn


class LinearRegressionPyTorch(nn.Module):
    """PyTorch linear regression model converted from scikit-learn."""

    def __init__(self, input_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LogisticRegressionPyTorch(nn.Module):
    """PyTorch logistic regression model for both binary and multilabel classification."""

    def __init__(self, input_dim: int, output_dim: int = 1, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_binary = output_dim == 1
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

        # For binary classification, use Sigmoid
        # For multilabel, use Softmax
        self.activation = nn.Sigmoid() if self.is_binary else nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)

        # For binary classification, squeeze the output to match expected shape
        if self.is_binary:
            logits = logits.squeeze(-1)

        probabilities = self.activation(logits)
        return probabilities

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        with torch.no_grad():
            probabilities = self.forward(x)

            if self.is_binary:
                predictions = (probabilities >= threshold).float()
            else:
                # For multilabel, take the argmax
                predictions = torch.argmax(probabilities, dim=1)

        return predictions
