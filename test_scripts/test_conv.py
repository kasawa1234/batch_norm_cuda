import torch
import time
import torch.nn as nn
import cppcuda_bn


def bn2d_backward(grad_output, normalized, gamma, std_eps):
    N, C, H, W = grad_output.shape

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
gamma = torch.ones(4, device=device)
beta = torch.zeros(4, device=device)
BN = nn.BatchNorm2d(4, device=device, affine=False)


# forward
normalized_python = BN(a)

b = cppcuda_bn.bn_forward_conv(a, gamma, beta)
std_eps_cppcuda = b[-1, :, 0, 0].contiguous()
normalized_cppcuda = b[: -1, :, :, :]

c = cppcuda_bn.bn_forward_conv_parallel(a, gamma, beta)
std_eps_cppcuda_parallel = c[-1, :, 0, 0].contiguous()
normalized_cppcuda_parallel = c[: -1, :, :, :]

d = cppcuda_bn.bn_forward_conv_flatten(a, gamma, beta)
std_eps_cppcuda_flatten = d[-1, :, 0, 0].contiguous()
normalized_cppcuda_flatten = d[: -1, :, :, :]

e = cppcuda_bn.bn_forward_conv_sram(a, gamma, beta)
std_eps_cppcuda_sram = (e[-1, :, 0, 0] / (shape[0] * shape[2] * shape[3])).contiguous()
normalized_cppcuda_sram = e[: -1, :, :, :]


print("______________________________ normalized ______________________________")
print(abs(normalized_cppcuda - normalized_python).max())
print(abs(normalized_cppcuda_parallel - normalized_python).max())
print(abs(normalized_cppcuda_flatten - normalized_python).max())
print(abs(normalized_cppcuda_sram - normalized_python).max())

print("______________________________  std_eps  _______________________________")
print(abs(std_eps_cppcuda_parallel - std_eps_cppcuda).max())
print(abs(std_eps_cppcuda_flatten - std_eps_cppcuda).max())
print(abs(std_eps_cppcuda_sram - std_eps_cppcuda).max())


# backward
grad_output = torch.randn(shape, device=device)
grad_input_python, grad_gamma_python, grad_beta_python = bn2d_backward(grad_output, normalized_python, gamma, std_eps_cppcuda)


backward_cpp = cppcuda_bn.bn_backward_conv(grad_output, normalized_cppcuda, gamma, std_eps_cppcuda)
grad_input_cpp = backward_cpp[: -2, :, :, :]
grad_gamma_cpp = backward_cpp[-2, :, 0, 0]
grad_beta_cpp = backward_cpp[-1, :, 0, 0]


backward_cpp_parallel = cppcuda_bn.bn_backward_conv_parallel(grad_output, normalized_cppcuda_parallel, gamma, std_eps_cppcuda_parallel)
grad_input_cpp_parallel = backward_cpp_parallel[: -2, :, :, :]
grad_gamma_cpp_parallel = backward_cpp_parallel[-2, :, 0, 0]
grad_beta_cpp_parallel = backward_cpp_parallel[-1, :, 0, 0]


print("______________________________ grad_input ______________________________")
print(abs(grad_input_cpp - grad_input_python).max())
print(abs(grad_input_cpp_parallel - grad_input_python).max())

print("______________________________ grad_gamma ______________________________")
print(grad_gamma_cpp - grad_gamma_python)
print(grad_gamma_cpp_parallel - grad_gamma_python)

print("______________________________ grad_beta _______________________________")
print(grad_beta_cpp - grad_beta_python)
print(grad_beta_cpp_parallel - grad_beta_python)
