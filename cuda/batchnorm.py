from torch import nn
from torch.autograd import Function
import torch

import batchnorm_cuda

torch.manual_seed(42)


class BatchNorm2dFunction(Function):
    @staticmethod
    def forward(ctx, input, gamma, beta):
        outputs = batchnorm_cuda.batchnorm2d_forward(input, gamma, beta)
        variables = outputs[:-1]
        # print(outputs[0])
        # print(outputs[3])
        # print(outputs[-1])
        ctx.save_for_backward(*variables)

        return outputs[-1]

    @staticmethod
    def backward(ctx, d_output):
        outputs = batchnorm_cuda.batchnorm2d_backward(
            d_output.contiguous(), *ctx.saved_variables
        )
        d_input, d_gamma, d_beta = outputs
        # sd_input, d_gamma, d_beta = 0, 0, 0
        return d_input, d_gamma, d_beta


class BatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.bata = nn.Parameter(torch.zeros(num_features))

    def forward(self, input, state):
        return BatchNorm2dFunction.apply(input, self.gamma, self.beta, *state)
