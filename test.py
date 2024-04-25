import torch
import torch.nn as nn
import cppcuda_bn

a = torch.randn([256, 4], device='cuda:0')
b = cppcuda_bn.bn_forward_mlp(a, torch.ones(4, device='cuda:0'), torch.zeros(4, device='cuda:0'))
print(nn.BatchNorm1d(4, device='cuda:0', affine=False)(a))
print(b)
