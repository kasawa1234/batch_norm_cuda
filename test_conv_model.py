import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cppcuda_bn
import torch.optim as optim
from torch.autograd import Function
import time

class BatchNorm2dParallelFunction(Function):
    @staticmethod
    def forward(ctx, x, gamma, beta):
        y = cppcuda_bn.bn_forward_conv_parallel(x, gamma, beta)
        output = y[:-1, :, :, :]
        ctx.save_for_backward(x, y, gamma)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y, gamma = ctx.saved_tensors
        x_normalized = y[:-1, :, :, :]
        std = y[-1, :, 0, 0]
        std = std.contiguous()  # To make std contiguous in memory
        backward = cppcuda_bn.bn_backward_conv_parallel(grad_output, x_normalized, gamma, std)
        grad_input = backward[:-2, :, :, :]
        grad_gamma = backward[-2, :, 0, 0]
        grad_beta = backward[-1:, :, 0, 0]
        return grad_input, grad_gamma, grad_beta
    
class BatchNorm2dLoopFunction(Function):
    @staticmethod
    def forward(ctx, x, gamma, beta):
        y = cppcuda_bn.bn_forward_conv(x, gamma, beta)
        output = y[:-1, :, :, :]
        ctx.save_for_backward(x, y, gamma)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y, gamma = ctx.saved_tensors
        x_normalized = y[:-1, :, :, :]
        std = y[-1, :, 0, 0]
        std = std.contiguous()  # To make std contiguous in memory
        backward = cppcuda_bn.bn_backward_conv(grad_output, x_normalized, gamma, std)
        grad_input = backward[:-2, :, :, :]
        grad_gamma = backward[-2, :, 0, 0]
        grad_beta = backward[-1:, :, 0, 0]
        return grad_input, grad_gamma, grad_beta

class BatchNormParallel2d(nn.Module):
    def __init__(self, num_features):
        super(BatchNormParallel2d, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        if self.training:
            # 在训练模式下，运行BatchNormFunction的前向传播并计算新的running值
            y = BatchNorm2dParallelFunction.apply(x, self.gamma, self.beta)
            
        else:
            # 在评估模式下，直接使用running_mean和running_var进行归一化
            # y = self.gamma * (x - self.running_mean) / torch.sqrt(self.running_var + self.eps) + self.beta
            y = BatchNorm2dParallelFunction.apply(x, self.gamma, self.beta)
        return y
    
class BatchNormLoop2d(nn.Module):
    def __init__(self, num_features):
        super(BatchNormLoop2d, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        if self.training:
            # 在训练模式下，运行BatchNormFunction的前向传播并计算新的running值
            y = BatchNorm2dLoopFunction.apply(x, self.gamma, self.beta)
            
        else:
            # 在评估模式下，直接使用running_mean和running_var进行归一化
            # y = self.gamma * (x - self.running_mean) / torch.sqrt(self.running_var + self.eps) + self.beta
            y = BatchNorm2dLoopFunction.apply(x, self.gamma, self.beta)
        return y
    
class Conv_python(nn.Module):
    def __init__(self):
        super(Conv_python, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.bn1 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.bn2 = nn.BatchNorm2d(64)  
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  
        self.fc2 = nn.Linear(128, 10)  

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  
        x = x.view(-1, 64 * 7 * 7)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Conv_Parallel(nn.Module):
    def __init__(self):
        super(Conv_Parallel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 卷积层1
        self.bn1 = BatchNormParallel2d(32)  # 批归一化层1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 卷积层2
        self.bn2 = BatchNormParallel2d(64)  # 批归一化层2
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 全连接层1
        self.fc2 = nn.Linear(128, 10)  # 全连接层2，输出10个类别

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 2)  # 池化层
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 2)  # 池化层
        x = x.view(-1, 64 * 7 * 7)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Conv_Loop(nn.Module):
    def __init__(self):
        super(Conv_Loop, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 卷积层1
        self.bn1 = BatchNormLoop2d(32)  # 批归一化层1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 卷积层2
        self.bn2 = BatchNormLoop2d(64)  # 批归一化层2
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 全连接层1
        self.fc2 = nn.Linear(128, 10)  # 全连接层2，输出10个类别

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 2)  # 池化层
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 2)  # 池化层
        x = x.view(-1, 64 * 7 * 7)  # 展平层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

device = 'cuda:0'
# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化网络、损失函数和优化器
model_conv_python = Conv_python().to(device)
print(model_conv_python)
model_conv_loop = Conv_Loop().to(device)
print(model_conv_loop)
model_conv_parallel = Conv_Parallel().to(device)
print(model_conv_parallel)

criterion = nn.CrossEntropyLoss()
optimizer_conv_python = optim.Adam(model_conv_python.parameters(), lr=0.001)
optimizer_conv_loop = optim.Adam(model_conv_loop.parameters(), lr=0.001)
optimizer_conv_parallel = optim.Adam(model_conv_parallel.parameters(), lr=0.001)
epochs = 5


# train model
def train(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_correct_train, total_train = 0, 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            _, predicted_train = torch.max(output.data, 1)
            total_train += target.size(0)
            total_correct_train += (predicted_train == target).sum().item()
        accuracy_train = total_correct_train / total_train * 100
        print('Epoch {}: Loss: {:.6f}, Train Accuracy: {:.2f}%'.format(epoch + 1, loss.item(), accuracy_train))

# test model
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total * 100
    print('Test Accuracy: {:.2f}%'.format(accuracy))

start_conv_python_train = time.time()
train(model_conv_python, train_loader=train_loader, optimizer = optimizer_conv_python, criterion = criterion, epochs=epochs)
end_conv_python_train = time.time()

start_loop_train = time.time()
train(model_conv_loop, train_loader=train_loader, optimizer = optimizer_conv_loop, criterion = criterion, epochs=epochs)
end_loop_train = time.time()

start_parallel_train = time.time()
train(model_conv_parallel, train_loader=train_loader, optimizer = optimizer_conv_parallel, criterion = criterion, epochs=epochs)
end_parallel_train = time.time()

start_conv_python_test = time.time()
test(model_conv_python, test_loader=test_loader)
end_conv_python_test = time.time()

start_loop_test = time.time()
test(model_conv_loop, test_loader=test_loader)
end_loop_test = time.time()

start_parallel_test = time.time()
test(model_conv_parallel, test_loader = test_loader)
end_parallel_test = time.time()



print(end_conv_python_train - start_conv_python_train, end_loop_train - start_loop_train, end_parallel_train - start_parallel_train)
print(end_conv_python_test - start_conv_python_test, end_loop_test - start_loop_test, end_parallel_test - start_parallel_test)