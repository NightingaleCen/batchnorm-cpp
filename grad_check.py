from __future__ import division
from __future__ import print_function

import argparse
import os
import torch
from torch.autograd import gradcheck

parser = argparse.ArgumentParser()
parser.add_argument("example", choices=["py", "cpp", "cuda"])
parser.add_argument("-b", "--batch-size", type=int, default=5)
parser.add_argument("-f", "--feature_size", type=int, default=3)
parser.add_argument("-H", "--height", type=int, default=40)
parser.add_argument("-W", "--width", type=int, default=40)
parser.add_argument("-c", "--cuda", action="store_true")
options = parser.parse_args()

if options.example == "py":
    from python.lltm_baseline import LLTMFunction
elif options.example == "cpp":
    from cpp.lltm import LLTMFunction
else:
    from cuda.batchnorm import BatchNorm2dFunction

    options.cuda = True

device = torch.device("cuda") if options.cuda else torch.device("cpu")

kwargs = {"dtype": torch.float64, "device": device, "requires_grad": True}

input = torch.ones(
    options.batch_size, options.feature_size, options.height, options.width, **kwargs
)
gamma = torch.zeros(options.feature_size, **kwargs)
beta = torch.ones(options.feature_size, **kwargs)

variables = [input, gamma, beta]

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if gradcheck(BatchNorm2dFunction.apply, variables):
    print("Ok")
