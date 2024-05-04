import torch
import time
import torch.nn as nn
import cppcuda_bn


device = 'cuda:0'

a = torch.randn([256, 4, 15, 300], device=device)
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

start_cppcuda_sram = time.time()
d = cppcuda_bn.bn_forward_conv_sram(a, gamma, beta)
end_cppcuda_sram = time.time()

# print("normalized_python: ", normalized_python)
# print("normalized_cppcuda:", normalized_cppcuda)
print(c[: -1, :, :, :] - normalized_python)

print(end_python - start_python, end_cppcuda - start_cppcuda, end_cppcuda_parallel - start_cppcuda_parallel, end_cppcuda_sram - start_cppcuda_sram, sep='\n')


std_eps = b[-1, :, 0, 0]
std_eps = std_eps.contiguous()  # To make std_eps contiguous in memory
