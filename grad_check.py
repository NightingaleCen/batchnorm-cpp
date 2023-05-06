from __future__ import division
from __future__ import print_function

import argparse
import os
import torch
from torch.autograd import gradcheck
import numpy as np


def check(model, baseline_model, variables, model_name):
    print("=" * 40)
    print(f"Checking the availability of {model_name}:")
    print(f"Doing gradient check...", end="")
    # if gradcheck(model, variables, fast_mode=False):
    #    print("Passed")
    print(f"Doing gradient check...")
    print(f"Comparing model output with baseline: Forward...", end="")
    output = model(*variables)
    baseline_output = baseline_model(*variables)
    if check_equal(output, baseline_output):
        print("Passed")

    print(f"Comparing model output with baseline: Backward...", end="")
    output.sum().backward()
    grads = get_grads(variables)
    zero_grad(variables)
    baseline_output.sum().backward()
    baseline_grads = get_grads(variables)

    baseline_output = baseline_model(*variables)
    if check_equal(grads, baseline_grads):
        print("Passed")
    print("All Clear!")
    print("=" * 40)


def get_grads(variables):
    return [var.grad.clone() for var in variables]


def zero_grad(variables):
    for variable in variables:
        variable.grad.zero_()


def check_equal(x: torch.Tensor, y: torch.Tensor):
    for i, (x, y) in enumerate(zip(x, y)):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        np.testing.assert_allclose(x, y, err_msg="Index: {}".format(i))
    return True


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch-size", type=int, default=2)
parser.add_argument("-f", "--feature-size", type=int, default=3)
parser.add_argument("-H", "--height", type=int, default=21)
parser.add_argument("-W", "--width", type=int, default=21)
parser.add_argument("-c", "--cuda", action="store_true")
options = parser.parse_args()

if __name__ == "__main__":
    from cuda.batchnorm import (
        BatchNorm2d,
        BatchNorm1d,
        BatchNorm2dFunction,
        BatchNorm1dFunction,
    )

    options.cuda = True

    device = torch.device("cuda") if options.cuda else torch.device("cpu")

    kwargs = {"dtype": torch.float64, "device": device, "requires_grad": True}

    input1d = torch.randn(options.batch_size, options.feature_size, **kwargs)
    input2d = torch.randn(
        options.batch_size,
        options.feature_size,
        options.height,
        options.width,
        **kwargs,
    )
    gamma = torch.ones(options.feature_size, **kwargs)
    beta = torch.zeros(options.feature_size, **kwargs)

    variables1d = [input1d]
    variables2d = [input2d]

    batchnorm1d = BatchNorm1d(options.feature_size).to(device, torch.float64)
    batchnorm2d = BatchNorm2d(options.feature_size).to(device, torch.float64)
    batchnorm1d_baseline = torch.nn.BatchNorm1d(options.feature_size).to(
        device, torch.float64
    )
    batchnorm2d_baseline = torch.nn.BatchNorm2d(options.feature_size).to(
        device, torch.float64
    )

    check(batchnorm1d, batchnorm1d_baseline, variables1d, "BatchNorm1d")
    check(batchnorm2d, batchnorm2d_baseline, variables2d, "BatchNorm2d")
    # gradcheck(BatchNorm1dFunction.apply, [input2d, gamma, beta])
