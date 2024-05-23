import numpy as np
import torch
import torch.nn as nn

import cppcuda_bn
from utils import show_time, show_time_backward, bn1d_backward


device = 'cuda:0'
shape = [256, 8]

a = torch.randn(shape, device=device)
grad_output = torch.randn(shape, device=device)
gamma = torch.ones(shape[1], device=device)
beta = torch.zeros(shape[1], device=device)
BN = nn.BatchNorm1d(shape[1], device=device, affine=False)

b = cppcuda_bn.bn_forward_mlp(a, gamma, beta)
normalized_cppcuda = b[: -1, :]
std_eps = b[-1, :]

c = cppcuda_bn.bn_forward_mlp_sram(a, gamma, beta)
normalized_cppcuda_sram = c[: -1, :]
std_eps_sram_N = c[-1, :]


print("Running pytorch...")
cuda_time, _ = show_time(BN, a, gamma, beta, type='torch')
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

print("Running mlp_naive...")
cuda_time, _ = show_time(cppcuda_bn.bn_forward_mlp, a, gamma, beta)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

print("Running mlp_sram...")
cuda_time, _ = show_time(cppcuda_bn.bn_forward_mlp_sram, a, gamma, beta)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

print("Running mlp_backward_python...")
cuda_time, _ = show_time_backward(bn1d_backward, grad_output, normalized_cppcuda_sram, gamma, std_eps)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

print("Running mlp_backward_naive...")
cuda_time, _ = show_time_backward(cppcuda_bn.bn_backward_mlp, grad_output, normalized_cppcuda_sram, gamma, std_eps)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

print("Running mlp_backward_sram...")
cuda_time, _ = show_time_backward(cppcuda_bn.bn_backward_mlp_sram, grad_output, normalized_cppcuda_sram, gamma, std_eps_sram_N)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
