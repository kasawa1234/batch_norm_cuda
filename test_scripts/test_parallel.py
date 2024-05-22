import torch
import time
import torch.nn as nn
import cppcuda_bn


def bn2d_backward(grad_output, normalized, gamma, std_eps):
    
    N,C,H,W = grad_output.shape

    # 梯度的均值和方差的计算
    grad_gamma = torch.sum(grad_output * normalized, dim=[0, 2, 3], keepdim=True)
    grad_beta = torch.sum(grad_output, dim=[0, 2, 3], keepdim=True)

    # dx_ 是对输入数据的梯度的中间变量
    dx_ = gamma.view(1, C, 1, 1) * grad_output

    grad_input = N * H * W * dx_ - dx_.sum(dim=[0, 2, 3], keepdim = True) - normalized * (dx_ * normalized).sum(dim=[0, 2, 3], keepdim=True)
    grad_input /= (N * H * W * std_eps.view(1, C, 1, 1))

    # 返回梯度
    return grad_input, grad_gamma.flatten(), grad_beta.flatten()


device = 'cuda:0'
shape = [256, 4, 32, 32]

a = torch.randn(shape, device=device)
print(a[:, 0, :, :].mean(), a[:, 1, :, :].mean(), a[:, 2, :, :].mean(), a[:, 3, :, :].mean())
gamma = torch.ones(4, device=device)
beta = torch.zeros(4, device=device)
BN = nn.BatchNorm2d(4, device=device, affine=False)


# forward
normalized_python = BN(a)

b = cppcuda_bn.bn_forward_conv(a, gamma, beta)
std_eps_cppcuda = b[-1, :, 0, 0]
normalized_cppcuda = b[: -1, :, :, :]

c = cppcuda_bn.bn_forward_conv_parallel(a, gamma, beta)
std_eps_cppcuda_parallel = c[-1, :, 0, 0]
normalized_cppcuda_parallel = c[: -1, :, :, :]
std_eps = std_eps_cppcuda_parallel.contiguous()  # To make std_eps contiguous in memory

d = cppcuda_bn.bn_forward_conv_sram(a, gamma, beta)
std_eps_cppcuda_sram = d[-1, :, 0, 0]
normalized_cppcuda_sram = d[: -1, :, :, :]


# print("normalized_python: ", normalized_python)
# print("normalized_cppcuda:", normalized_cppcuda)
print((normalized_cppcuda - normalized_python).max())
print((normalized_cppcuda - normalized_cppcuda_parallel).max())
print((normalized_cppcuda - normalized_cppcuda_sram).max())
print((std_eps_cppcuda_parallel - std_eps_cppcuda).max())


# backward
grad_output = torch.randn(shape, device=device)
grad_input_python, grad_gamma_python, grad_beta_python = bn2d_backward(grad_output, normalized_python, gamma, std_eps)


backward_cpp = cppcuda_bn.bn_backward_conv(grad_output, normalized_cppcuda, gamma, std_eps)
grad_input_cpp = backward_cpp[: -2, :, :, :]
grad_gamma_cpp = backward_cpp[-2, :, 0, 0]
grad_beta_cpp = backward_cpp[-1, :, 0, 0]


backward_cpp_parallel = cppcuda_bn.bn_backward_conv_parallel(grad_output, normalized_cppcuda, gamma, std_eps)
grad_input_cpp_parallel = backward_cpp[: -2, :, :, :]
grad_gamma_cpp_parallel = backward_cpp[-2, :, 0, 0]
grad_beta_cpp_parallel = backward_cpp[-1, :, 0, 0]

# print("grad_input_python:           ", grad_input_python)
# print("grad_input_cppcuda:          ", grad_input_cpp)
# print("grad_input_cppcuda_parallel: ", grad_input_cpp_parallel)

# print("grad_gamma_python:           ", grad_gamma_python)
# print("grad_gamma_cppcuda:          ", grad_gamma_cpp)
# print("grad_gamma_cppcuda_parallel: ", grad_gamma_cpp_parallel)

# print("grad_beta_python:            ", grad_beta_python)
# print("grad_beta_cppcuda:           ", grad_beta_cpp)
# print("grad_beta_cppcuda_parallel:  ", grad_beta_cpp_parallel)

# print(grad_input_cpp_parallel - grad_input_python)
print(grad_gamma_cpp_parallel - grad_gamma_python)
print(grad_beta_cpp_parallel - grad_beta_python)
