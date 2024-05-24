import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import cppcuda_bn
from utils import show_time, show_time_backward, bn1d_backward


device = 'cuda:0'
shape = [256, 8]

a = torch.randn(shape, device=device)
grad_output = torch.randn(shape, device=device)
gamma = torch.ones(shape[1], device=device)
beta = torch.zeros(shape[1], device=device)
BN = nn.BatchNorm1d(shape[1], device=device, affine=False)

b = cppcuda_bn.bn_forward_mlp(a, gamma, beta)
normalized_cppcuda = b[: -1, :]
std_eps = b[-1, :]

c = cppcuda_bn.bn_forward_mlp_sram(a, gamma, beta)
normalized_cppcuda_sram = c[: -1, :]
std_eps_sram_N = c[-1, :]


print("Running pytorch...")
cuda_time, _ = show_time(BN, a, gamma, beta, type='torch')
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
pytorch_times = cuda_time

print("Running mlp_naive...")
cuda_time, _ = show_time(cppcuda_bn.bn_forward_mlp, a, gamma, beta)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
mlp_naive_times = cuda_time

print("Running mlp_sram...")
cuda_time, _ = show_time(cppcuda_bn.bn_forward_mlp_sram, a, gamma, beta)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
mlp_sram_times = cuda_time

print("Running mlp_backward_python...")
cuda_time, _ = show_time_backward(bn1d_backward, grad_output, normalized_cppcuda_sram, gamma, std_eps)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
mlp_back_python_times = cuda_time

print("Running mlp_backward_naive...")
cuda_time, _ = show_time_backward(cppcuda_bn.bn_backward_mlp, grad_output, normalized_cppcuda_sram, gamma, std_eps)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
mlp_back_naive_times = cuda_time

print("Running mlp_backward_sram...")
cuda_time, _ = show_time_backward(cppcuda_bn.bn_backward_mlp_sram, grad_output, normalized_cppcuda_sram, gamma, std_eps_sram_N)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
mlp_back_sram_times = cuda_time


data = [pytorch_times, mlp_naive_times, mlp_sram_times]

# 每个类型的名称
labels = ['PyTorch', 'MLP Naive', 'MLP SRAM']

# 每个类型的测试次数
num_tests = len(pytorch_times)

# 生成x轴的位置
x = np.arange(len(labels))

# 设置柱状图的宽度
width = 0.05

# 创建绘图对象
fig, ax = plt.subplots(figsize=(8, 4))

# 在柱状图中添加数据
for i in range(num_tests):
    ax.bar(x + i * width, [data[j][i] for j in range(len(labels))], width, label=f'Test {i+1}')

# 设置x轴的刻度为标签位置
ax.set_xticks(x + width * (num_tests - 1) / 2)
ax.set_xticklabels(labels)

# 添加标签和标题
ax.set_xlabel('Type')
ax.set_ylabel('Time (us)')
ax.set_title('MLP Forward Time Comparison')

# 添加图例
ax.legend()

# 保存图形为图片文件
plt.savefig('save_figs/mlp_forward.png')

# 关闭图形以释放内存
plt.close()


data = [mlp_back_python_times, mlp_back_naive_times, mlp_back_sram_times]

# 每个类型的名称
labels = ['PyTorch', 'MLP Naive', 'MLP SRAM']

# 每个类型的测试次数
num_tests = len(pytorch_times)

# 生成x轴的位置
x = np.arange(len(labels))

# 设置柱状图的宽度
width = 0.05

# 创建绘图对象
fig, ax = plt.subplots(figsize=(8, 4))

# 在柱状图中添加数据
for i in range(num_tests):
    ax.bar(x + i * width, [data[j][i] for j in range(len(labels))], width, label=f'Test {i+1}')

# 设置x轴的刻度为标签位置
ax.set_xticks(x + width * (num_tests - 1) / 2)
ax.set_xticklabels(labels)

# 添加标签和标题
ax.set_xlabel('Type')
ax.set_ylabel('Time (us)')
ax.set_title('MLP Backward Time Comparison')

# 添加图例
ax.legend()

# 保存图形为图片文件
plt.savefig('save_figs/mlp_backward.png')

# 关闭图形以释放内存
plt.close()
