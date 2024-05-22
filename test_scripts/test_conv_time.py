import numpy as np
import torch
import torch.nn as nn

import cppcuda_bn
from sync import show_time


device = 'cuda:0'
shape = [256, 8, 32, 32]

a = torch.randn(shape, device=device)
gamma = torch.ones(shape[1], device=device)
beta = torch.zeros(shape[1], device=device)
BN = nn.BatchNorm2d(shape[1], device=device, affine=False)


print("Running pytorch...")
cuda_time, _ = show_time(BN, a, gamma, beta, type='torch')
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

print("Running conv_naive...")
cuda_time, _ = show_time(cppcuda_bn.bn_forward_conv, a, gamma, beta)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

print("Running conv_parallel...")
cuda_time, _ = show_time(cppcuda_bn.bn_forward_conv_parallel, a, gamma, beta)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

print("Running conv_flatten...")
cuda_time, _ = show_time(cppcuda_bn.bn_forward_conv_flatten, a, gamma, beta)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

print("Running conv_sram...")
cuda_time, _ = show_time(cppcuda_bn.bn_forward_conv_sram, a, gamma, beta)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
