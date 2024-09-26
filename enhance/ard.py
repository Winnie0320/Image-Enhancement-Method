import os
import cv2
import numpy as np
import math
from PIL import Image
import torch
import torchvision.transforms as transforms


# 定义最大灰度级数
gray_level = 16
average = 0.5
input_folder = './input/'
output_folder = './output/'

comp_list = []
def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    # print("图像的高宽分别为：height,width", height, width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    # print("max_gray_level:", max_gray_level)
    return max_gray_level + 1

#彩色图像
def ard(image):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # 假设图片是 .jpg 或 .png 格式
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.tiff'):
                image_path = os.path.join(root, file)
                complexity = test(image_path)
                if complexity is not None:
                    print(f'Image {file} complexity: {complexity}')
                if complexity < average:
                    clip_limit = width/4

                else:
                    clip_limit = width/8

                lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_channel = lab_image[:, :, 0]
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                enhanced_l_channel = clahe.apply(l_channel)
                enhanced_lab_image = lab_image.copy()
                enhanced_lab_image[:, :, 0] = enhanced_l_channel
                enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
                return enhanced_image



image_files = os.listdir(input_folder)
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    enhanced_image = ard(image)
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, enhanced_image)

print('Image enhancement complete！')

