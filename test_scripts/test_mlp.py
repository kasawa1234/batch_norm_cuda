import torch
import torch.nn as nn
import cppcuda_bn

from utils import bn1d_backward


device = "cuda:0"
shape = [256, 4]


a = torch.randn(shape, device=device)
grad_output = torch.randn(shape, device=device)
gamma = torch.ones(shape[1], device=device)
beta = torch.zeros(shape[1], device=device)

normalized_python = nn.BatchNorm1d(shape[1], device=device, affine=False)(a)

b = cppcuda_bn.bn_forward_mlp(a, gamma, beta)
normalized_cppcuda = b[: -1, :]
std_eps = b[-1, :]

c = cppcuda_bn.bn_forward_mlp_sram(a, gamma, beta)
normalized_cppcuda_sram = c[: -1, :]
std_eps_sram_N = c[-1, :]
std_eps_sram = std_eps_sram_N / shape[0]


print(abs(normalized_cppcuda - normalized_python).max())
print(abs(normalized_cppcuda_sram - normalized_python).max())

print(abs(std_eps_sram - std_eps).max())


grad_input_python, grad_gamma_python, grad_beta_python = bn1d_backward(grad_output, normalized_python, gamma, std_eps)

backward_cppcuda = cppcuda_bn.bn_backward_mlp(grad_output, normalized_cppcuda, gamma, std_eps)
grad_input_cppcuda = backward_cppcuda[: -2, :]
grad_gamma_cppcuda = backward_cppcuda[-2, :]
grad_beta_cppcuda = backward_cppcuda[-1, :]

backward_cppcuda_sram = cppcuda_bn.bn_backward_mlp_sram(grad_output, normalized_cppcuda_sram, gamma, std_eps_sram_N)
grad_input_cppcuda_sram = backward_cppcuda_sram[: -2, :]
grad_gamma_cppcuda_sram = backward_cppcuda_sram[-2, :]
grad_beta_cppcuda_sram = backward_cppcuda_sram[-1, :]


print(abs(grad_input_cppcuda - grad_input_python).max())
print(abs(grad_input_cppcuda_sram - grad_input_python).max())

print(abs(grad_gamma_cppcuda - grad_gamma_python).max())
print(abs(grad_gamma_cppcuda_sram - grad_gamma_python).max())

print(abs(grad_beta_cppcuda - grad_beta_python).max())
print(abs(grad_beta_cppcuda_sram - grad_beta_python).max())

