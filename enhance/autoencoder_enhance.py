import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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
        # 隐藏层有吗

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

# 定义损失函数
def loss(y_true, y_pred):
    # 计算输入图像和重建图像的平均亮度
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)

    loss = torch.abs(mean_true - mean_pred)
    return loss

# 定义优化器
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)


# 训练模型
def train_model(model, train_loader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for data in train_loader:
            images, _ = data
            optimizer.zero_grad()
            outputs = model(images)
            loss = total_loss(images, outputs)  # 修正此处调用
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 定义总的损失函数，包括自编码器的重建损失和亮度保持的损失项
def total_loss(y_true, y_pred):
    return 0.5*nn.MSELoss()(y_true, y_pred) + 0.5*loss(y_true, y_pred)

# 训练模型
train_model(autoencoder, train_loader, optimizer)

torch.save(autoencoder.state_dict(), 'autoencoder_model.pth')

