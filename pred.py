import torch
import torchvision
from torchvision import transforms

# 定義預處理轉換
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 載入測試資料集
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False)

# 載入模型權重
model = YourModel()  # 替換成你的模型類別
model.load_state_dict(torch.load(
    '/path/to/weight/folder/your_model_weights.pth'))

# 設定模型為評估模式
model.eval()

# 定義計算正確率的函式


def compute_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy


# 驗證模型並計算最終分數
total_accuracy = 0
total_samples = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        accuracy = compute_accuracy(outputs, labels)

        total_accuracy += accuracy * labels.size(0)
        total_samples += labels.size(0)

final_score = total_accuracy / total_samples
print(f"Final score: {final_score}")
