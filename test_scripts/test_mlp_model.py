import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Function
import cppcuda_bn
import time

class BatchNorm1dFunction(Function):
    @staticmethod
    def forward(ctx, x, gamma, beta):

        y = cppcuda_bn.bn_forward_mlp(x, gamma, beta)
        output = y[:-1, :]
        ctx.save_for_backward(x, y, gamma)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y, gamma = ctx.saved_tensors
        x_normalized = y[:-1, :]
        std = y[-1, :]
        std = std.contiguous()  # To make std contiguous in memory
        backward = cppcuda_bn.bn_backward_mlp(grad_output, x_normalized, gamma, std)
        grad_input = backward[:-2, :]
        grad_gamma = backward[-2, :]
        grad_beta = backward[-1:, :]
        return grad_input, grad_gamma, grad_beta
    
class BatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        if self.training:
            # 在训练模式下，运行BatchNormFunction的前向传播并计算新的running值
            y = BatchNorm1dFunction.apply(x, self.gamma, self.beta)
            
        else:
            # 在评估模式下，直接使用running_mean和running_var进行归一化
            # y = self.gamma * (x - self.running_mean) / torch.sqrt(self.running_var + self.eps) + self.beta
            y = BatchNorm1dFunction.apply(x, self.gamma, self.beta)
        return y

class MLP_Python(nn.Module):
    def __init__(self):
        super(MLP_Python, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.bn1 = nn.BatchNorm1d(512)  
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)  
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # 将图像展平为一维向量
        x = x.view(-1, 28 * 28)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
    
class MLP_Cuda(nn.Module):
    def __init__(self):
        super(MLP_Cuda, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.bn1 = BatchNorm1d(512)  
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = BatchNorm1d(256)  
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # 将图像展平为一维向量
        x = x.view(-1, 28 * 28)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
    
device = 'cuda:0'
# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model_mlp_python = MLP_Python().to(device)
model_mlp_cuda = MLP_Cuda().to(device)


criterion = nn.CrossEntropyLoss()
optimizer_python = optim.Adam(model_mlp_python.parameters(), lr=0.001)
optimizer_cuda = optim.Adam(model_mlp_cuda.parameters(), lr=0.001)

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

start_python_train = time.time()
train(model_mlp_python, train_loader=train_loader, optimizer = optimizer_python, criterion = criterion, epochs=epochs)
end_python_train = time.time()

start_cuda_train = time.time()
train(model_mlp_cuda, train_loader=train_loader, optimizer = optimizer_cuda, criterion = criterion, epochs=epochs)
end_cuda_train = time.time()

start_python_test = time.time()
test(model_mlp_python, test_loader)
end_python_test = time.time()

start_cuda_test = time.time()
test(model_mlp_cuda, test_loader)
end_cuda_test = time.time()

print(end_python_train - start_python_train, end_cuda_train - start_cuda_train)
print(end_python_test - start_python_test, end_cuda_test - start_cuda_test)

