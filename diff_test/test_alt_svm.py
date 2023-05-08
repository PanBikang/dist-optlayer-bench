import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# 定义超参数
batch_size = 64
learning_rate = 0.01
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/storage/data/panbk/dataset/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/storage/data/panbk/dataset/', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

# 定义在线软间隔支持向量机模型
class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        return x

    def hinge_loss(self, output, target):
        batch_size = output.shape[0]
        correct_scores = output[range(batch_size), target]
        # print(f"correct_scores.shape: {correct_scores.shape}")
        margins = torch.clamp(output - correct_scores.view(-1, 1) + 1, min=0)
        margins[range(batch_size), target] = 0
        loss = margins.mean()
        return loss

class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        return x

    def hinge_loss(self, output, target):
        batch_size = output.shape[0]
        correct_scores = output[range(batch_size), target]
        # print(f"correct_scores.shape: {correct_scores.shape}")
        margins = torch.clamp(output - correct_scores.view(-1, 1) + 1, min=0)
        margins[range(batch_size), target] = 0
        loss = margins.mean()
        return loss
    
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        return x

# 初始化模型、优化器和损失函数
model = SVM().to(device)
model1 = LinearModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
criterion1 = nn.CrossEntropyLoss()

# 训练模型
for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.hinge_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('SVM--Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer1.zero_grad()
        output = model1(data)
        loss = criterion1(output, target)
        loss.backward()
        optimizer1.step()
        if batch_idx % 100 == 0:
            print('FC--Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# 测试模型
model.eval()
model1.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print('\nSVM--Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), accuracy))


with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model1(data)
        test_loss += criterion1(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print('\nFC--Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), accuracy))
