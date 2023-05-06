import math
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(42)


class BatchNorm1d(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, input: torch.Tensor):
        mean = input.mean(dim=0, keepdim=True)
        sigma2 = torch.pow(input - mean, 2).mean(dim=0, keepdim=True)
        normalized_input = (input - mean) * torch.rsqrt(sigma2 + 1e-5)
        output = self.gamma * normalized_input + self.beta

        return output
