import torch
import torch.nn as nn
import cppcuda_bn
import numpy as np

import time


device = "cuda:0"
iteration = 10
time_python = 0
time_conv = 0
time_conv_parallel = 0
time_conv_sram = 0

for i in range(iteration):
    # shape_conv = [
    #     np.random.randint(1, 257),
    #     np.random.randint(1, 17),
    #     np.random.randint(1, 513),
    #     np.random.randint(1, 513)
    # ]
    shape_conv = [256, 8, 32, 32]
    a = torch.randn(shape_conv, device=device)
    gamma = torch.ones(shape_conv[1], device=device)
    beta = torch.zeros(shape_conv[1], device=device)
    BN = nn.BatchNorm2d(shape_conv[1], device=device, affine=False)

    start_time_python = time.time()
    normalized_python = BN(a)
    end_time_python = time.time()

    start_cppcuda = time.time()
    b = cppcuda_bn.bn_forward_conv(a, gamma, beta)
    end_cppcuda = time.time()

    start_cppcuda_parallel = time.time()
    c = cppcuda_bn.bn_forward_conv_parallel(a, gamma, beta)
    end_cppcuda_parallel = time.time()

    start_cppcuda_sram = time.time()
    d = cppcuda_bn.bn_forward_conv_sram(a, gamma, beta)
    end_cppcuda_sram = time.time()

    if i != 0:
        time_python += end_time_python - start_time_python
        time_conv += end_cppcuda - start_cppcuda
        time_conv_parallel += end_cppcuda_parallel - start_cppcuda_parallel
        time_conv_sram += end_cppcuda_sram - start_cppcuda_sram
    print(end_time_python - start_time_python, end_cppcuda - start_cppcuda, end_cppcuda_parallel - start_cppcuda_parallel, end_cppcuda_sram - start_cppcuda_sram)

print(time_python, time_conv, time_conv_parallel, time_conv_sram)
