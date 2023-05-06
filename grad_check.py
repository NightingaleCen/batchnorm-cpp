from __future__ import division
from __future__ import print_function

import argparse
import os
import torch
from torch.autograd import gradcheck

parser = argparse.ArgumentParser()
parser.add_argument("example", choices=["py", "cpp", "cuda"])
parser.add_argument("-b", "--batch-size", type=int, default=2)
parser.add_argument("-f", "--feature_size", type=int, default=3)
parser.add_argument("-H", "--height", type=int, default=21)
parser.add_argument("-W", "--width", type=int, default=21)
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

# input = torch.tensor([-4, -3, -2, -1, 1, 2, 3, 4], **kwargs).reshape(
#    (options.batch_size, options.feature_size, options.height, options.width)
# )
input = torch.randn(
    options.batch_size, options.feature_size, options.height, options.width, **kwargs
)
gamma = torch.ones(options.feature_size, **kwargs)
beta = torch.zeros(options.feature_size, **kwargs)

variables = [input, gamma, beta]
# print(input)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
"""print(input[:, 0, :, :])
print(input[:, 0, :, :])
print(((input[:, 0, :, :] - input[:, 0, :, :].mean()) ** 2).mean())
print(input[:, 0, :, :].var())"""

m = torch.nn.BatchNorm2d(options.feature_size, dtype=torch.float64, device=device)
m = m(input)
n = BatchNorm2dFunction.apply(input, gamma, beta)
# print(m, "\n", n)
print(torch.equal(m, n))
print((m - n).max())


grad = torch.randn_like(input)
# grad = torch.arange(
#    options.batch_size * options.feature_size * options.height * options.width, **kwargs
# ).reshape((options.batch_size, options.feature_size, options.height, options.width))
output_grad_n = torch.autograd.grad(n, input, grad_outputs=grad)[0]
output_grad_m = torch.autograd.grad(m, input, grad_outputs=grad)[0]
# print(output_grad_m, "\n", output_grad_n)
print(torch.equal(output_grad_m, output_grad_n))
print((output_grad_m - output_grad_n).max())

# TODO:
# MLP输入batchnorm（另开一个文件）
# baseline（pytorch手写，pytorch官方）
# 正确性测试代码（gradcheck，正反向结果对照）
# 速度测试代码

if gradcheck(BatchNorm2dFunction.apply, variables, fast_mode=False):
    print("Passed")
