import torch
import time


def show_time(func, a, gamma, beta, type='cppcuda', ntest=10):
    """
    Parameters:
    - func: ``PyTorch`` Class or self implemented function
    - a, gamma, beta: BN parameters
    - type: \'cppcuda\' or \'torch\'
    - ntest: default = 10, test epochs
    """
    times = list()
    res = list()
    # GPU warm up
    for _ in range(10):
        if type == 'torch':
            func(a)
        elif type == 'cppcuda':
            func(a, gamma, beta)
        else:
            raise ValueError(f"Expected \'torch\' or \'cppcuda\', but got {type}")
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        if type == 'torch':
            r = func(a)
        elif type == 'cppcuda':
            r = func(a, gamma, beta)
        else:
            raise ValueError(f"Expected \'torch\' or \'cppcuda\', but got {type}")
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()

        times.append((end_time-start_time)*1e6)
        res.append(r)
    return times, res

def show_time_backward(func, grad_output, normalized, gamma, std_eps, ntest=10):
    """
    Parameters:
    - func: ``PyTorch`` Class or self implemented function
    - grad_output, normalized, gamma, std_eps: BN parameters
    - ntest: default = 10, test epochs
    """
    times = list()
    res = list()
    # GPU warm up
    for _ in range(10):
        func(grad_output, normalized, gamma, std_eps)
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        r = func(grad_output, normalized, gamma, std_eps)
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()

        times.append((end_time-start_time)*1e6)
        res.append(r)
    return times, res

def bn1d_backward(grad_output, normalized, gamma: torch.Tensor, std_eps, device='cuda:0'):
    N, _ = grad_output.shape
    
    # 计算输入和参数的梯度
    gamma = gamma.unsqueeze(0)

    grad_gamma = (grad_output * normalized).sum(dim=0, keepdim=True)
    grad_beta = grad_output.sum(dim=0, keepdim=True)
    
    dx_ = torch.mm(torch.ones(N, 1, device=device), gamma) * grad_output
    grad_input = N * dx_ - dx_.sum(dim=0) - normalized * (dx_ * normalized).sum(dim=0)
    grad_input /= (N * std_eps)
    
    # 返回梯度和更新的滑动平均
    return grad_input, grad_gamma.flatten(), grad_beta.flatten()

def bn2d_backward(grad_output, normalized, gamma, std_eps, device='cuda:0'):
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
