import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


batch_size = 1000
folder = './data'

train_data = pd.read_csv(folder + '/train.csv')
test_data = pd.read_csv(folder + '/test.csv')

# print(train_data.shape)
# print(test_data.shape)


# Splitting train dataset into X and Y.Normalizing it by dividing it with 255

x = train_data.iloc[:, 1:].values / 255
y = train_data.iloc[:, 0].values


# visulaizing numbers in our dataset.
r = 4
c = 6
fig = plt.figure(figsize=(r, c), dpi=100)
for i in range(1, r*c+1):
    img = x[i].reshape(28, 28)
    ax = fig.add_subplot(r, c, i)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.title.set_text(y[i])

    plt.imshow(img, cmap='gray')
plt.savefig('./fig/show.png')


train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2)


# converting our data into a datloader object.

train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
train_y = torch.from_numpy(train_y).type(torch.LongTensor)

val_x = torch.from_numpy(val_x).type(torch.FloatTensor)
val_y = torch.from_numpy(val_y).type(torch.LongTensor)


trn = TensorDataset(train_x, train_y)
val = TensorDataset(val_x, val_y)

trn = DataLoader(trn, batch_size=1000)
val = DataLoader(val, batch_size=1000)

print(train_x[0].shape)


# 未正規化的資料
train_data_unnorm = pd.read_csv(folder + '/train.csv')
x_unnorm = train_data_unnorm.iloc[:, 1:].values  # 未進行除以255的操作
y_unnorm = train_data_unnorm.iloc[:, 0].values

# 拆分未正規化的資料
train_x_unnorm, val_x_unnorm, train_y_unnorm, val_y_unnorm = train_test_split(
    x_unnorm, y_unnorm, test_size=0.2)

# 轉換為 Tensor
train_x_unnorm = torch.from_numpy(train_x_unnorm).type(torch.FloatTensor)
train_y_unnorm = torch.from_numpy(train_y_unnorm).type(torch.LongTensor)
val_x_unnorm = torch.from_numpy(val_x_unnorm).type(torch.FloatTensor)
val_y_unnorm = torch.from_numpy(val_y_unnorm).type(torch.LongTensor)

# 創建 DataLoader
trn_unnorm = DataLoader(TensorDataset(
    train_x_unnorm, train_y_unnorm), batch_size=1000)
val_unnorm = DataLoader(TensorDataset(
    val_x_unnorm, val_y_unnorm), batch_size=1000)


# 假設您的測試數據已經是一個Numpy數組，並且已經被正規化
test_x = test_data.values / 255  # 假設test_data是您的測試數據DataFrame

# 將數據轉換成Tensor
test_x = torch.from_numpy(test_x).type(torch.FloatTensor)

test_dataset = TensorDataset(test_x)  # 假設 test_y 包含了測試數據的標籤
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 檢查 DataLoader 返回的第一個批次
for target in test_loader:
    print(f"Target shape: {target.shape}")
    break  # 只檢查第一個批次
