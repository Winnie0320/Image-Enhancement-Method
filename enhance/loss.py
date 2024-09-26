import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 加载数据并进行预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

# 初始化自编码器模型
autoencoder = Autoencoder()

# 定义亮度保持的损失函数
def brightness_preservation_loss(y_true, y_pred):
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    loss = torch.abs(mean_true - mean_pred)
    return loss

# 定义总的损失函数，包括自编码器的重建损失和亮度保持的损失项
def total_loss(y_true, y_pred):
    return 0.5 * nn.MSELoss()(y_true, y_pred) + 0.5 * brightness_preservation_loss(y_true, y_pred)

# 定义优化器
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# 训练模型并记录损失
def train_model(model, train_loader, optimizer, num_epochs=10):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in train_loader:
            images, _ = data
            optimizer.zero_grad()
            outputs = model(images)
            loss = total_loss(images, outputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
        avg_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    return train_losses

# 训练模型并获取损失
train_losses = train_model(autoencoder, train_loader, optimizer)

# 绘制损失图
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# 保存模型
torch.save(autoencoder.state_dict(), 'autoencoder_model.pth')

# 评估模型
def evaluate_model(model, val_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for data in val_loader:
            images, _ = data
            outputs = model(images)
            loss = total_loss(images, outputs)
            total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(val_loader.dataset)
        print(f'Validation Loss: {avg_loss:.4f}')

# 评估模型
evaluate_model(autoencoder, val_loader)
