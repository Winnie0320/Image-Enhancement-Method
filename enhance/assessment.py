import os
import cv2
import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import lpips
from skimage.metrics import structural_similarity as ssim

import torchvision.transforms as transforms
from lpips import LPIPS



# 初始化结果列表
psnr_list = []
ssim_list = []
ev_list = []
lpips_list = []


def calculate_ev(input_image):
    # 计算梯度图
    sobelx = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_map = np.sqrt(np.square(sobelx) + np.square(sobely))
    # 对梯度图进行阈值处理
    threshold = 50  # 根据实际情况调整阈值
    gradient_map = cv2.threshold(gradient_map, threshold, 255, cv2.THRESH_BINARY)[1]
    # 计算边缘可见度
    ev = np.sum(gradient_map > 0) / (input_image.shape[0] * input_image.shape[1])
    return ev
def calculate_ssim(input_image, output_image):
    original_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

    ssim_value, _ = ssim(original_gray, enhanced_gray, full=True)
    return ssim_value

def calculate_lpips(input_image, output_image):
    loss_fn = LPIPS(net='alex')

    # 加载图像并进行预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image1_tensor = transform(input_image).unsqueeze(0)
    image2_tensor = transform(output_image).unsqueeze(0)

    # 计算LPIPS相似度
    with torch.no_grad():
        lpips_value = loss_fn(image1_tensor, image2_tensor).item()

    return lpips_value


# 定义输入输出文件夹路径
input_folder = './input/'
output_folder = './output/'


# 获取输入文件夹中所有的图片文件名
image_files = os.listdir(input_folder)

# 遍历每张图片进行计算
for image_file in image_files:
    # 读入原始图片和增强后的图片
    input_image_path = os.path.join(input_folder, image_file)
    output_image_path = os.path.join(output_folder, image_file)
    input_image = cv2.imread(input_image_path)
    output_image = cv2.imread(output_image_path)

    # 计算PSNR
    psnr = peak_signal_noise_ratio(input_image, output_image)
    psnr_list.append(psnr)

    # 计算SSIM
    ssimvalue = calculate_ssim(input_image, output_image)
    ssim_list.append(ssimvalue)

    # 计算LPIPS
    lpipsvalue = calculate_lpips(input_image, output_image)
    lpips_list.append(lpipsvalue)


# 计算平均值
mean_psnr = np.mean(psnr_list)
mean_ssim = np.mean(ssim_list)
mean_lpips = np.mean(lpips_list)


print(f'PSNR平均值: {mean_psnr}')
print(f'SSIM平均值: {mean_ssim}')
print(f'LPIPS平均值: {mean_lpips}')
