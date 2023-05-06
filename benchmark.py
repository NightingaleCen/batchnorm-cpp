from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch

TIME_SCALES = {"s": 1, "ms": 1000, "us": 1000000}


def evaluate(model, variables, model_name):
    # Force CUDA initialization
    output = model(*variables)
    output.sum().backward()
    forward_min = math.inf

    forward_time = 0
    backward_min = math.inf
    backward_time = 0
    for _ in range(options.runs):
        batchnorm1d.zero_grad()

        start = time.time()
        output = model(*variables)
        elapsed = time.time() - start
        forward_min = min(forward_min, elapsed)
        forward_time += elapsed

        start = time.time()
        output.sum().backward()
        elapsed = time.time() - start
        backward_min = min(backward_min, elapsed)
        backward_time += elapsed

    scale = TIME_SCALES[options.scale]
    forward_min *= scale
    backward_min *= scale
    forward_average = forward_time / options.runs * scale
    backward_average = backward_time / options.runs * scale

    print(
        "{0} Forward: {1:.3f}/{2:.3f} {5} | Backward {3:.3f}/{4:.3f} {5}".format(
            model_name,
            forward_min,
            forward_average,
            backward_min,
            backward_average,
            options.scale,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("example", choices=["py", "cpp", "cuda"])
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-f", "--feature-size", type=int, default=3)
    parser.add_argument("-H", "--height", type=int, default=21)
    parser.add_argument("-W", "--width", type=int, default=21)
    parser.add_argument("-r", "--runs", type=int, default=100)
    parser.add_argument("--scale", choices=["s", "ms", "us"], default="us")
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("-d", "--double", action="store_true")
    options = parser.parse_args()

    if options.example == "py":
        from python.batchnorm import BatchNorm1d
    elif options.example == "cpp":
        from cpp.lltm import LLTM
    else:
        from cuda.batchnorm import BatchNorm1d, BatchNorm2d

        options.cuda = True

    device = torch.device("cuda") if options.cuda else torch.device("cpu")
    dtype = torch.float64 if options.double else torch.float32

    kwargs = {"dtype": dtype, "device": device, "requires_grad": True}
    input1d = torch.randn(options.batch_size, options.feature_size, **kwargs)
    input2d = torch.randn(
        options.batch_size,
        options.feature_size,
        options.height,
        options.width,
        **kwargs
    )
    variable1d = [input1d]
    variable2d = [input2d]

    batchnorm1d = BatchNorm1d(options.feature_size).to(device, dtype)
    # batchnorm2d = BatchNorm2d(options.feature_size).to(device, dtype)
    batchnorm1d_torch = torch.nn.BatchNorm1d(options.feature_size).to(device, dtype)
    # batchnorm2d_torch = torch.nn.BatchNorm2d(options.feature_size).to(device, dtype)

    evaluate(batchnorm1d, variable1d, "BatchNorm1d")
    # evaluate(batchnorm2d, variable2d, "BatchNorm2d")
    evaluate(batchnorm1d_torch, variable1d, "Pytorch BatchNorm1d")
    # evaluate(batchnorm2d_torch, variable2d, "Pytorch BatchNorm2d")
