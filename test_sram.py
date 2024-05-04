import torch
import torch.nn as nn
import cppcuda_bn

import time


device = "cuda:0"

a = torch.randn([256, 4], device=device)
grad_output = torch.randn([256, 4], device=device)
gamma = torch.ones(4, device=device)
beta = torch.zeros(4, device=device)

start_time_python = time.time()
normalized_python = nn.BatchNorm1d(4, device=device, affine=False)(a)
end_time_python = time.time()

start_time_cppcuda = time.time()
b = cppcuda_bn.bn_forward_mlp(a, gamma, beta)
end_time_cppcuda = time.time()

normalized_cppcuda = b[: -1, :]
std_eps = b[-1, :]

start_time_cppcuda_sram = time.time()
c = cppcuda_bn.bn_forward_mlp_sram(a, gamma, beta)
end_time_cppcuda_sram = time.time()

normalized_cppcuda_sram = c[: -1, :]
std_eps_sram = c[-1, :]


print("normalized_python:       ", normalized_python)
print("normalized_cppcuda:      ", normalized_cppcuda)
print("normalized_cppcuda_sram: ", normalized_cppcuda_sram)

print(end_time_python - start_time_python, end_time_cppcuda - start_time_cppcuda, end_time_cppcuda_sram - start_time_cppcuda_sram)
