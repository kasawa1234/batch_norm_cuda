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
    return grad_input, grad_gamma, grad_beta


device = 'cuda:0'
shape = [256, 4, 32, 300]

a = torch.randn(shape, device=device)
gamma = torch.ones(4, device=device)
beta = torch.zeros(4, device=device)
BN = nn.BatchNorm2d(4, device=device, affine=False)

start_python = time.time()
normalized_python = BN(a)
end_python = time.time()

start_cppcuda = time.time()
b = cppcuda_bn.bn_forward_conv(a, gamma, beta)
end_cppcuda = time.time()

start_cppcuda_parallel = time.time()
c = cppcuda_bn.bn_forward_conv_parallel(a, gamma, beta)
end_cppcuda_parallel = time.time()


# print("normalized_python: ", normalized_python)
# print("normalized_cppcuda:", normalized_cppcuda)
# print(c[: -1, :, :, :] - normalized_python)


std_eps = c[-1, :, 0, 0]
normalized_cppcuda = c[: -1, :, :, :]
std_eps = std_eps.contiguous()  # To make std_eps contiguous in memory

grad_output = torch.randn(shape, device=device)

start_python_back = time.time()
grad_input_python, grad_gamma_python, grad_beta_python = bn2d_backward(grad_output, normalized_python, gamma, std_eps)
end_python_back = time.time()

start_cppcuda_back = time.time()
backward_cpp = cppcuda_bn.bn_backward_conv(grad_output, normalized_cppcuda, gamma, std_eps)
end_cppcuda_back = time.time()
grad_input_cpp = backward_cpp[: -2, :, :, :]
grad_gamma_cpp = backward_cpp[-2, :, 0, 0]
grad_beta_cpp = backward_cpp[-1, :, 0, 0]

start_cppcuda_parallel_back = time.time()
backward_cpp_parallel = cppcuda_bn.bn_backward_conv_parallel(grad_output, normalized_cppcuda, gamma, std_eps)
end_cppcuda_parallel_back = time.time()
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

print(end_python - start_python, end_cppcuda - start_cppcuda, end_cppcuda_parallel - start_cppcuda_parallel)
print(end_python_back - start_python_back, end_cppcuda_back - start_cppcuda_back, end_cppcuda_parallel_back - start_cppcuda_parallel_back)

