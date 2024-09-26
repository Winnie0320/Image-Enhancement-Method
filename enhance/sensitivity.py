import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# 读取数据
file_path = 'output.txt'  # 替换为实际的文件路径

# 尝试读取数据并清洗
try:
    data = pd.read_csv(file_path, delimiter=',', header=None)

    # 将数据转换为数值，忽略无法转换的值
    data = data.apply(pd.to_numeric, errors='coerce')

    # 输出原始数据和清洗后的数据行数
    print("原始数据行数:", len(data))

    # 移除包含NaN的行
    data.dropna(inplace=True)

    # 输出清洗后的数据行数
    print("清洗后的数据行数:", len(data))

    # 检查是否有有效数据
    if data.shape[0] == 0:
        raise ValueError("清洗后的数据没有有效样本，无法进行线性回归。")

    # 数据准备
    X = data.iloc[:, :-1]  # 特征
    y = data.iloc[:, -1]  # 目标变量

    # 拟合线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 获取模型的系数
    coefficients = model.coef_


    # 计算灵敏度系数
    def calculate_sensitivity_coefficients(coefficients, X, y):
        X_mean = X.mean()
        y_mean = y.mean()
        sensitivity_coefficients = coefficients * (X_mean / y_mean)
        return sensitivity_coefficients


    # 计算灵敏度系数
    sensitivity_coefficients = calculate_sensitivity_coefficients(coefficients, X, y)

    # 输出灵敏度系数
    feature_names = [f'Feature_{i + 1}' for i in range(X.shape[1])]
    for feature, sensitivity in zip(feature_names, sensitivity_coefficients):
        print(f'{feature}: {sensitivity:.4f}')

except Exception as e:
    print(f"发生错误: {e}")
