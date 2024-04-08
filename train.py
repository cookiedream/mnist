import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt


num = "2_64_normalization"
# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義MLP模型

# -----------------修改-----------------


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
# -----------------修改-----------------


class MLP3(nn.Module):
    def __init__(self):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # 將圖像從28*28展平到512
        self.fc2 = nn.Linear(512, 512)  # 第二層512到512
        self.fc3 = nn.Linear(512, 10)  # 最後一層到10個類別
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)  # 展平圖像
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
# -----------------修改-----------------


class MLP4(nn.Module):
    def __init__(self):
        super(MLP4, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # 將圖像從28*28展平到512
        self.fc2 = nn.Linear(512, 512)  # 第二層512到512
        self.fc3 = nn.Linear(512, 256)  # 第三層512到256
        self.fc4 = nn.Linear(256, 10)  # 最後一層到10個類別
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)
# -----------------修改-----------------


class MLP5(nn.Module):
    def __init__(self):
        super(MLP5, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # 將圖像從28*28展平到512
        self.fc2 = nn.Linear(512, 512)  # 第二層512到512
        self.fc3 = nn.Linear(512, 256)  # 第三層512到256
        self.fc4 = nn.Linear(256, 128)  # 第四層256到128
        self.fc5 = nn.Linear(128, 10)  # 最後一層到10個類別
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)


# -----------------修改-----------------
# 數據加載
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 初始化模型、損失函數和優化器
model = MLP().to(device)
# model = MLP3().to(device)
# model = MLP4().to(device)
# -----------------修改-----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 訓練模型
epochs = 50
train_accuracy_history = []
test_accuracy_history = []
test_precision_history = []
test_recall_history = []
test_f1_history = []

for epoch in range(epochs):
    model.train()
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    train_accuracy_history.append(train_accuracy)

    # 測試模型
    model.eval()
    correct_test = 0
    total_test = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = 100 * correct_test / total_test
    test_accuracy_history.append(test_accuracy)

    # 計算精度、召回率和F1分數
    precision = precision_score(all_labels, all_predicted, average='macro')
    recall = recall_score(all_labels, all_predicted, average='macro')
    f1 = f1_score(all_labels, all_predicted, average='macro')

    test_precision_history.append(precision)
    test_recall_history.append(recall)
    test_f1_history.append(f1)

    # 打印每個epoch的結果
    print(f'Epoch [{epoch+1}/{epochs}], Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')

# 繪製訓練歷程圖
plt.figure(figsize=(12, 8))
plt.plot(train_accuracy_history, label='Train Accuracy')
plt.plot(test_accuracy_history, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.title('Training Process')
plt.legend()
plt.savefig(f'./finally/train_process{num}.png')
