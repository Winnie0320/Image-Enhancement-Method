import os
import torch
from torchvision import transforms
from PIL import Image
from ronghe import Autoencoder

# 定义输入和输出文件夹路径
input_folder = './input/'
output_folder = './output/'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 加载自编码器模型
autoencoder = Autoencoder()  # 假设 Autoencoder 类在同一文件中定义
autoencoder.load_state_dict(torch.load('autoencoder_model.pth'))  # 加载训练好的模型权重
autoencoder.eval()  # 将模型设置为评估模式，这样不会影响到BatchNorm层等

# 定义图像预处理，将图像转换为张量
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载并处理原始图像，然后保存增强后的图像
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.tiff')):
        # 加载原始图像
        img = Image.open(os.path.join(input_folder, filename))
        # 将图像转换为张量
        img_tensor = transform(img)
        # 使用自编码器模型进行图像增强
        enhanced_img_tensor = autoencoder(img_tensor.unsqueeze(0))  # 添加批量维度
        enhanced_img_tensor.squeeze_(0)  # 移除批量维度

        # # 调整通道顺序为 (H, W, 3)，以便正确地转换为 PIL 图像
        # enhanced_img_tensor_permuted = enhanced_img_tensor.permute(2, 1, 0)
        # # 将张量转换为 PIL 图像
        # to_pil = transforms.ToPILImage()
        # enhanced_img = to_pil(enhanced_img_tensor_permuted)

        # 将张量转换为PIL图像
        enhanced_img = transforms.ToPILImage()(enhanced_img_tensor)
        # 调整增强后的图像尺寸与原始图像一致
        # enhanced_img = enhanced_img.resize(img.size, Image.ANTIALIAS)
        enhanced_img = enhanced_img.resize(img.size, Image.LANCZOS)
        # 保存增强后的图像
        enhanced_img.save(os.path.join(output_folder, filename))

