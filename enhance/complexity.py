import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import MinMaxScaler

MIN_CONTRAST, MAX_CONTRAST = 0, 256
MIN_ENERGY, MAX_ENERGY = 0, 1
MIN_CORR, MAX_CORR = -1, 1
MIN_HOMOGENEITY, MAX_HOMOGENEITY = 0, 1

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)
def calculate_features(image_path):
    # 读取图像并转换为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 计算灰度共生矩阵
    glcm = graycomatrix(image, [1], [0], symmetric=True, normed=True)

    # 提取特征
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # 特征值归一化
    normalized_contrast = normalize(contrast, MIN_CONTRAST, MAX_CONTRAST)
    normalized_energy = normalize(energy, MIN_ENERGY, MAX_ENERGY)
    normalized_correlation = normalize(correlation, MIN_CORR, MAX_CORR)
    normalized_homogeneity = normalize(homogeneity, MIN_HOMOGENEITY, MAX_HOMOGENEITY)

    # 计算complexity
    complexity = 0.6 * normalized_contrast + 0.1 * normalized_energy + 0.2 * normalized_correlation + 0.1 * normalized_homogeneity

    return {
        'contrast': normalized_contrast,
        'energy': normalized_energy,
        'correlation': normalized_correlation,
        'homogeneity': normalized_homogeneity,
        'complexity': complexity
    }

def process_folder(folder_paths, output_file):
    with open(output_file, 'w') as f:
        for folder_path in folder_paths:
            if not isinstance(folder_path, str):
                raise ValueError("folder_paths should be a list of strings")
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, file)
                        features = calculate_features(image_path)


                        f.write(f"{features['complexity']:.5f}\n")

#

if __name__ == '__main__':
    # List of folder paths
    folder_paths = [
        '.../bicycle',
        '.../Boat',
        '.../bottle',
        '.../bus',
        '.../car',
        '.../cat',
        '.../chair',
        '.../cup',
        '.../dog',
        '.../Motorbike',
        '.../people',
        '.../Table'
    ]

    output_file = 'output.txt'

    plot_file = 'complexity_scatter_plot(normalize).png'

    # Process all folders and calculate complexity
    process_folder(folder_paths, output_file)



