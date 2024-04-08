import torch
import torch.nn.functional as F
import torch.nn as nn


class MLP_Large(nn.Module):
    def __init__(self):
        super(MLP_Large, self).__init__()
        self.linear1 = nn.Linear(784, 1024)  # 第一個隱藏層: 784 -> 1024
        self.linear2 = nn.Linear(1024, 128)  # 第二個隱藏層: 1024 -> 128
        self.linear3 = nn.Linear(128, 10)    # 輸出層: 128 -> 10

    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        output = F.log_softmax(X, dim=1)
        return output


class MLP_Medium(nn.Module):
    def __init__(self):
        super(MLP_Medium, self).__init__()
        self.linear1 = nn.Linear(784, 128)  # 第一個隱藏層: 784 -> 128
        self.linear2 = nn.Linear(128, 64)   # 第二個隱藏層: 128 -> 64
        self.linear3 = nn.Linear(64, 10)    # 輸出層: 64 -> 10

    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        output = F.log_softmax(X, dim=1)
        return output


class MLP_Small(nn.Module):
    def __init__(self):
        super(MLP_Small, self).__init__()
        self.linear1 = nn.Linear(784, 64)  # 第一個隱藏層: 784 -> 64
        self.linear2 = nn.Linear(64, 64)   # 第二個隱藏層: 64 -> 64
        self.linear3 = nn.Linear(64, 10)   # 輸出層: 64 -> 10

    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        output = F.log_softmax(X, dim=1)
        return output


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784, 250)
        self.linear2 = nn.Linear(250, 100)
        self.linear3 = nn.Linear(100, 10)

    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        output = F.log_softmax(X, dim=1)
        return output


class MLP3(nn.Module):
    def __init__(self):
        super(MLP3, self).__init__()
        self.linear1 = nn.Linear(784, 250)
        self.linear2 = nn.Linear(250, 150)
        self.linear3 = nn.Linear(150, 100)
        self.linear4 = nn.Linear(100, 10)

    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = F.relu(self.linear3(X))
        X = self.linear4(X)
        output = F.log_softmax(X, dim=1)
        return output


class MLP4(nn.Module):
    def __init__(self):
        super(MLP4, self).__init__()
        self.linear1 = nn.Linear(784, 350)
        self.linear2 = nn.Linear(350, 250)
        self.linear3 = nn.Linear(250, 150)
        self.linear4 = nn.Linear(150, 100)
        self.linear5 = nn.Linear(100, 10)

    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = F.relu(self.linear3(X))
        X = F.relu(self.linear4(X))
        X = self.linear5(X)
        output = F.log_softmax(X, dim=1)
        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class TwoLayerCNN(nn.Module):
    def __init__(self):
        super(TwoLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=1, padding=1)  # 正確的輸入通道數
        self.dropout = nn.Dropout(0.5)
        # 假設輸出類別數為10，這裡根據最後一個池化層的輸出調整全連接層的輸入大小
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)  # 确保扁平化后的特征向量形状正确
        x = self.fc(x)  # 正确的全连接层调用
        output = F.log_softmax(x, dim=1)
        return output


# ---------------------------------------------------------------------------------------------------------------------

# 三層CNN模型


class ThreeLayerCNN(nn.Module):
    def __init__(self):
        super(ThreeLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 注意：这里假设输入图像大小为28x28，并且经过3次池化操作后的特征图大小为3x3（这需要根据实际的卷积和池化层调整）
        self.fc = nn.Linear(128 * 3 * 3, 10)  # 更新全连接层的输入尺寸

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))  # Conv -> ReLU -> Pool
        x = F.relu(self.pool(self.conv2(x)))  # Conv -> ReLU -> Pool
        x = F.relu(self.pool(self.conv3(x)))  # Conv -> ReLU -> Pool
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        x = self.dropout2(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

# ---------------------------------------------------------------------------------------------------------------------

# 四層CNN模型


class FourLayerCNN(nn.Module):
    def __init__(self):
        super(FourLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 注意：这里假设输入图像大小为28x28，并且经过4次池化操作后的特征图大小需要根据实际调整
        self.fc = nn.Linear(256 * 2 * 2, 10)  # 假设经过4次池化后的特征图大小为2x2

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))  # Conv -> ReLU -> Pool
        x = F.relu(self.pool(self.conv2(x)))  # Conv -> ReLU -> Pool
        x = F.relu(self.pool(self.conv3(x)))  # Conv -> ReLU -> Pool
        x = F.relu(self.pool(self.conv4(x)))  # Conv -> ReLU -> Pool
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        x = self.dropout2(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output
