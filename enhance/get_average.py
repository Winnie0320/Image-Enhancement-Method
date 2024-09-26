import numpy as np
import re

def calculate_average_from_file(file_path):
    data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 移除大括号和其他多余的字符
                cleaned_line = re.sub(r'[{}]', '', line.strip())

                try:
                    value = float(cleaned_line)
                    data.append(value)
                except ValueError:
                    print(f"Invalid data value encountered: {cleaned_line}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    if data:
        average_value = np.average(data)
        return average_value
    else:
        print("No valid data found.")
        return None


if __name__ == '__main__':
    file_path = 'output.txt'  # 替换为你的文件路径

    average = calculate_average_from_file(file_path)
    if median is not None:
        print(f"Average value: {average}")
    else:
        print("Could not calculate average.")
