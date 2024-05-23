import numpy as np
import torch
import torch.nn as nn

import cppcuda_bn
from utils import show_time, show_time_backward, bn2d_backward


device = 'cuda:0'
shape = [256, 8, 32, 32]

a = torch.randn(shape, device=device)
grad_output = torch.randn(shape, device=device)
gamma = torch.ones(shape[1], device=device)
beta = torch.zeros(shape[1], device=device)
BN = nn.BatchNorm2d(shape[1], device=device, affine=False)

b = cppcuda_bn.bn_forward_conv(a, gamma, beta)
std_eps_cppcuda = b[-1, :, 0, 0].contiguous()
normalized_cppcuda = b[: -1, :, :, :]

c = cppcuda_bn.bn_forward_conv_parallel(a, gamma, beta)
std_eps_cppcuda_parallel = c[-1, :, 0, 0].contiguous()
normalized_cppcuda_parallel = c[: -1, :, :, :]

e = cppcuda_bn.bn_forward_conv_sram(a, gamma, beta)
std_eps_cppcuda_sram_N = e[-1, :, 0, 0].contiguous()
normalized_cppcuda_sram = e[: -1, :, :, :]


print("Running PyTorch...")
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

print("Running conv_backward_python...")
cuda_time, _ = show_time_backward(bn2d_backward, grad_output, normalized_cppcuda_sram, gamma, std_eps_cppcuda)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

print("Running conv_backward_naive...")
cuda_time, _ = show_time_backward(cppcuda_bn.bn_backward_conv, grad_output, normalized_cppcuda_sram, gamma, std_eps_cppcuda)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

print("Running conv_backward_parallel...")
cuda_time, _ = show_time_backward(cppcuda_bn.bn_backward_conv_parallel, grad_output, normalized_cppcuda_sram, gamma, std_eps_cppcuda_parallel)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

print("Running conv_backward_sram...")
cuda_time, _ = show_time_backward(cppcuda_bn.bn_backward_conv_sram, grad_output, normalized_cppcuda_sram, gamma, std_eps_cppcuda_sram_N)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
