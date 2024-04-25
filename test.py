import torch
import torch.nn as nn
import cppcuda_bn

device = "cuda:0"

def backward(grad_output, normalized, gamma: torch.Tensor, std_eps):
    N, _ = grad_output.shape
    
    # 计算输入和参数的梯度
    gamma = gamma.unsqueeze(0)

    grad_gamma = (grad_output * normalized).sum(dim=0, keepdim=True)
    grad_beta = grad_output.sum(dim=0, keepdim=True)
    
    dx_ = torch.mm(torch.ones(N, 1, device=device), gamma) * grad_output
    grad_input = N * dx_ - dx_.sum(dim=0) - normalized * (dx_ * normalized).sum(dim=0)
    grad_input /= (N * std_eps)

    # grad_input2 = N * grad_output - torch.mm(torch.ones(N, 1), grad_gamma) * normalized - torch.mm(torch.ones(N, 1), grad_beta)
    # grad_input2 = torch.mm(torch.ones(N, 1), gamma) * grad_input2
    # grad_input2 /= (N * np.sqrt(var_eps))
    
    # 返回梯度和更新的滑动平均
    return grad_input, grad_gamma, grad_beta

a = torch.randn([256, 4], device=device)
grad_output = torch.randn([256, 4], device=device)
gamma = torch.ones(4, device=device)
beta = torch.zeros(4, device=device)

normalized_python = nn.BatchNorm1d(4, device=device, affine=False)(a)

b = cppcuda_bn.bn_forward_mlp(a, gamma, beta)
normalized_cppcuda = b[: -1, :]
std_eps = b[-1, :]

print("normalized_python: ", normalized_python)
print("normalized_cppcuda:", normalized_cppcuda)

grad_input_python, grad_gamma_python, grad_beta_python = backward(grad_output, normalized_python, gamma, std_eps)
backward_cpp = cppcuda_bn.bn_backward_mlp(grad_output, normalized_cppcuda, gamma, std_eps)
grad_input_cpp = backward_cpp[: -2, :]
grad_gamma_cpp = backward_cpp[-2, :]
grad_beta_cpp = backward_cpp[-1, :]

print("grad_input_python:   ", grad_input_python)
print("grad_input_cppcuda:  ", grad_input_cpp)

print("grad_gamma_python:   ", grad_gamma_python)
print("grad_gamma_cppcuda:  ", grad_gamma_cpp)

print("grad_beta_python:    ", grad_beta_python)
print("grad_beta_cppcuda:   ", grad_beta_cpp)

