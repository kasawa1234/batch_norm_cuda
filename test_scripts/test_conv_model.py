import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cppcuda_bn
import torch.optim as optim
from torch.autograd import Function
import matplotlib.pyplot as plt
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
    
class BatchNorm2dParallelSramFunction(Function):
    @staticmethod
    def forward(ctx, x, gamma, beta):
        y = cppcuda_bn.bn_forward_conv_sram(x, gamma, beta)
        output = y[:-1, :, :, :]
        ctx.save_for_backward(x, y, gamma)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y, gamma = ctx.saved_tensors
        N, C, H, W = grad_output.shape
        x_normalized = y[:-1, :, :, :]
        std = y[-1, :, 0, 0] / (N * H * W)
        std = std.contiguous()  # To make std contiguous in memory
        backward = cppcuda_bn.bn_backward_conv_sram(grad_output, x_normalized, gamma, std)
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
            y = BatchNorm2dParallelFunction.apply(x, self.gamma, self.beta)
            
        else:
            y = BatchNorm2dParallelFunction.apply(x, self.gamma, self.beta)
        return y
    
class BatchNormParallelSram2d(nn.Module):
    def __init__(self, num_features):
        super(BatchNormParallelSram2d, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        if self.training:
            y = BatchNorm2dParallelSramFunction.apply(x, self.gamma, self.beta)
            
        else:
            y = BatchNorm2dParallelSramFunction.apply(x, self.gamma, self.beta)
        return y
    
class BatchNormLoop2d(nn.Module):
    def __init__(self, num_features):
        super(BatchNormLoop2d, self).__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        if self.training:
            y = BatchNorm2dLoopFunction.apply(x, self.gamma, self.beta)
            
        else:
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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.bn1 = BatchNormParallel2d(32)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.bn2 = BatchNormParallel2d(64)  
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  
        self.fc2 = nn.Linear(128, 10)  

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 2)  
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 2)  
        x = x.view(-1, 64 * 7 * 7)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Conv_Parallel_Sram(nn.Module):
    def __init__(self):
        super(Conv_Parallel_Sram, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.bn1 = BatchNormParallelSram2d(32)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.bn2 = BatchNormParallelSram2d(64)  
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  
        self.fc2 = nn.Linear(128, 10)  

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 2)  
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 2)  
        x = x.view(-1, 64 * 7 * 7)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Conv_Loop(nn.Module):
    def __init__(self):
        super(Conv_Loop, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  
        self.bn1 = BatchNormLoop2d(32)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.bn2 = BatchNormLoop2d(64)  
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  
        self.fc2 = nn.Linear(128, 10)  

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 2) 
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = F.max_pool2d(x, 2)  
        x = x.view(-1, 64 * 7 * 7)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

device = 'cuda:0'
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model_conv_python = Conv_python().to(device)
model_conv_loop = Conv_Loop().to(device)
model_conv_parallel = Conv_Parallel().to(device)
model_conv_parallel_sram = Conv_Parallel_Sram().to(device)

loss_list_python = []
loss_list_loop = []
loss_list_parallel = []
loss_list_parallel_sram = []

criterion = nn.CrossEntropyLoss()
optimizer_conv_python = optim.Adam(model_conv_python.parameters(), lr=0.001)
optimizer_conv_loop = optim.Adam(model_conv_loop.parameters(), lr=0.001)
optimizer_conv_parallel = optim.Adam(model_conv_parallel.parameters(), lr=0.001)
optimizer_conv_parallel_sram = optim.Adam(model_conv_parallel_sram.parameters(), lr=0.001)

epochs = 5


# train model
def train(model, train_loader, optimizer, criterion, epochs, loss_list):
    model.train()
    for epoch in range(epochs):
        total_correct_train, total_train = 0, 0
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted_train = torch.max(output.data, 1)
            total_train += target.size(0)
            total_correct_train += (predicted_train == target).sum().item()
        epoch_loss = total_loss / len(train_loader)
        loss_list.append(epoch_loss)
        accuracy_train = total_correct_train / total_train * 100
        print('Epoch {}: Loss: {:.6f}, Train Accuracy: {:.2f}%'.format(epoch + 1, epoch_loss, accuracy_train))

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

print("Running Pytorch Model.......")
torch.cuda.synchronize(device="cuda:0")
start_conv_python_train = time.time()
train(model_conv_python, train_loader=train_loader, optimizer = optimizer_conv_python, criterion = criterion, epochs=epochs,loss_list = loss_list_python)
torch.cuda.synchronize(device="cuda:0")
end_conv_python_train = time.time()

print("Running Edition 1 Model.......")
torch.cuda.synchronize(device="cuda:0")
start_loop_train = time.time()
train(model_conv_loop, train_loader=train_loader, optimizer = optimizer_conv_loop, criterion = criterion, epochs=epochs, loss_list = loss_list_loop)
torch.cuda.synchronize(device="cuda:0")
end_loop_train = time.time()

print("Running Conv Parallel Model.......")
torch.cuda.synchronize(device="cuda:0")
start_parallel_train = time.time()
train(model_conv_parallel, train_loader=train_loader, optimizer = optimizer_conv_parallel, criterion = criterion, epochs=epochs, loss_list = loss_list_parallel)
torch.cuda.synchronize(device="cuda:0")
end_parallel_train = time.time()

print("Running Conv Parallel Sram Model.......")
torch.cuda.synchronize(device="cuda:0")
start_parallel_sram_train = time.time()
train(model_conv_parallel_sram, train_loader=train_loader, optimizer = optimizer_conv_parallel_sram, criterion = criterion, epochs=epochs, loss_list = loss_list_parallel_sram)
torch.cuda.synchronize(device="cuda:0")
end_parallel_sram_train = time.time()

# print("Running Pytorch Model Test.......")
# torch.cuda.synchronize(device="cuda:0")
# start_conv_python_test = time.time()
# test(model_conv_python, test_loader=test_loader)
# torch.cuda.synchronize(device="cuda:0")
# end_conv_python_test = time.time()

# print("Running Edition1 Model Test.......")
# torch.cuda.synchronize(device="cuda:0")
# start_loop_test = time.time()
# test(model_conv_loop, test_loader=test_loader)
# torch.cuda.synchronize(device="cuda:0")
# end_loop_test = time.time()

# print("Running Conv Parallel Model Test.......")
# torch.cuda.synchronize(device="cuda:0")
# start_parallel_test = time.time()
# test(model_conv_parallel, test_loader = test_loader)
# torch.cuda.synchronize(device="cuda:0")
# end_parallel_test = time.time()

# print("Running Conv Parallel Sram Model Test.......")
# torch.cuda.synchronize(device="cuda:0")
# start_parallel_sram_test = time.time()
# test(model_conv_parallel_sram, test_loader = test_loader)
# torch.cuda.synchronize(device="cuda:0")
# end_parallel_sram_test = time.time()

def plot_combined_loss(loss_list1, loss_list2, loss_list3, loss_list4, model_names):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list1, label=model_names[0], marker='o')
    plt.plot(loss_list2, label=model_names[1], marker='s')
    plt.plot(loss_list3, label=model_names[2], marker='^')
    plt.plot(loss_list4, label=model_names[3], marker='D')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch for All Models')
    plt.legend()
    plt.savefig('Combined_Training_Loss.png')
    plt.show()

def plot_training_time(times, models):
    plt.figure(figsize=(10, 6))
    bar_plot = plt.bar(models, times, color=['blue', 'green', 'red', 'yellow'])
    for i, v in enumerate(times):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time for Different Models')
    plt.xticks(models)
    plt.savefig('Training_Time_Conv.png')
    plt.show()

plot_combined_loss(loss_list_python, loss_list_loop, loss_list_parallel, loss_list_parallel_sram, ['Pytorch Model', 'Naive Model', 'Conv Parallel Model', 'Conv Parallel Sram Model'])
train_times = [
    end_conv_python_train - start_conv_python_train,
    end_loop_train - start_loop_train,
    end_parallel_train - start_parallel_train,
    end_parallel_sram_train - start_parallel_sram_train
]
model_names = ['Pytorch Model', 'Conv Naive Model', 'Conv Parallel Model', 'Conv Parallel Sram Model']
plot_training_time(train_times, model_names)

print(" Pytorch Model Train Time:{:.6f} s".format(end_conv_python_train - start_conv_python_train))
print(" Edition 1 Model Train Time:{:.6f} s".format(end_loop_train - start_loop_train))
print(" Conv Parallel Model Train Time:{:.6f} s".format(end_parallel_train - start_parallel_train))
print(" Conv Parallel Sram Model Train Time:{:.6f} s".format(end_parallel_sram_train - start_parallel_sram_train))

# print(" Pytorch Model Test Time:{:.6f} s".format(end_conv_python_test - start_conv_python_test))
# print(" Edition 1 Model Test Time:{:.6f} s".format(end_loop_test - start_loop_test))
# print(" Conv Parallel Model Test Time:{:.6f} s".format(end_parallel_test - start_parallel_test))
# print(" Conv Parallel Sram Model Test Time:{:.6f} s".format(end_parallel_sram_test - start_parallel_sram_test))

