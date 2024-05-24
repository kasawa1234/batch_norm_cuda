import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import cppcuda_bn
from utils import show_time, show_time_backward, bn2d_backward


device = 'cuda:0'
shape = [256, 8, 32, 32]

a = torch.randn(shape, device=device)
grad_output = torch.randn(shape, device=device)
gamma = torch.ones(shape[1], device=device)
beta = torch.zeros(shape[1], device=device)
BN = nn.BatchNorm2d(shape[1], device=device, affine=False)

b = cppcuda_bn.bn_forward_conv(a, gamma, beta)
std_eps_cppcuda = b[-1, :, 0, 0].contiguous()
normalized_cppcuda = b[: -1, :, :, :]

c = cppcuda_bn.bn_forward_conv_parallel(a, gamma, beta)
std_eps_cppcuda_parallel = c[-1, :, 0, 0].contiguous()
normalized_cppcuda_parallel = c[: -1, :, :, :]

e = cppcuda_bn.bn_forward_conv_sram(a, gamma, beta)
std_eps_cppcuda_sram_N = e[-1, :, 0, 0].contiguous()
normalized_cppcuda_sram = e[: -1, :, :, :]


print("Running PyTorch...")
cuda_time, _ = show_time(BN, a, gamma, beta, type='torch')
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
pytorch_times = cuda_time

print("Running conv_naive...")
cuda_time, _ = show_time(cppcuda_bn.bn_forward_conv, a, gamma, beta)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
conv_naive_times = cuda_time

print("Running conv_parallel...")
cuda_time, _ = show_time(cppcuda_bn.bn_forward_conv_parallel, a, gamma, beta)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
conv_parallel_times = cuda_time

print("Running conv_flatten...")
cuda_time, _ = show_time(cppcuda_bn.bn_forward_conv_flatten, a, gamma, beta)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
conv_flatten_times = cuda_time

print("Running conv_sram...")
cuda_time, _ = show_time(cppcuda_bn.bn_forward_conv_sram, a, gamma, beta)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
conv_sram_times = cuda_time

print("Running conv_backward_python...")
cuda_time, _ = show_time_backward(bn2d_backward, grad_output, normalized_cppcuda_sram, gamma, std_eps_cppcuda)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
conv_back_python_times = cuda_time

print("Running conv_backward_naive...")
cuda_time, _ = show_time_backward(cppcuda_bn.bn_backward_conv, grad_output, normalized_cppcuda_sram, gamma, std_eps_cppcuda)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
conv_back_naive_times = cuda_time

print("Running conv_backward_parallel...")
cuda_time, _ = show_time_backward(cppcuda_bn.bn_backward_conv_parallel, grad_output, normalized_cppcuda_sram, gamma, std_eps_cppcuda_parallel)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
conv_back_parallel_times = cuda_time

print("Running conv_backward_sram...")
cuda_time, _ = show_time_backward(cppcuda_bn.bn_backward_conv_sram, grad_output, normalized_cppcuda_sram, gamma, std_eps_cppcuda_sram_N)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
conv_back_sram_times = cuda_time


data = [pytorch_times, conv_naive_times, conv_parallel_times, conv_flatten_times, conv_sram_times]

# 每个类型的名称
labels = ['PyTorch', 'Conv Naive', 'Conv Parallel', 'Conv Flatten', 'Conv SRAM']

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
ax.set_title('Conv Forward Time Comparison')

# 添加图例
ax.legend()

# 保存图形为图片文件
plt.savefig('save_figs/conv_forward.png')

# 关闭图形以释放内存
plt.close()


data = [conv_back_python_times, conv_back_naive_times, conv_back_parallel_times, conv_back_sram_times]

# 每个类型的名称
labels = ['PyTorch', 'Conv Naive', 'Conv Parallel', 'Conv SRAM']

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
ax.set_title('Conv Backward Time Comparison')

# 添加图例
ax.legend()

# 保存图形为图片文件
plt.savefig('save_figs/conv_backward.png')

# 关闭图形以释放内存
plt.close()