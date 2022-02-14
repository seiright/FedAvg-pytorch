import torch
import torch.nn as nn
import torch.nn.functional as F


class Mnist_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor

"""定义CNN网络"""
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        """
        in_channels(int) – 输入信号的通道数目
        out_channels(int) – 卷积产生的通道数目
        kerner_size(int or tuple) - 卷积核的尺寸
        stride(int or tuple, optional) - 卷积步长
        padding(int or tuple, optional) - 输入的每一条边补充0的层数
        """
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)  # 卷积层 输出32*28*28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 池化层 2*2矩阵 输出32*14*14
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)  # 卷积层 输出64*14*14
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 池化层 2*2矩阵 输出64*7*7
        self.fc1 = nn.Linear(7*7*64, 512)  # 全连接层1：输入7*7*41，输出512
        self.fc2 = nn.Linear(512, 10)  # 全连接层2: 输入512，输出10

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

