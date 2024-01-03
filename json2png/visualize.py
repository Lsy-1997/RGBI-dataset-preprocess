import numpy as np
from PIL import Image
import random
import os

def random_color():
    """生成随机颜色"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def visualize_segmentation(png_path, color_map, output_path=None):
    """可视化语义分割标注文件"""
    # 读取标注文件
    mask = Image.open(png_path)
    mask = np.array(mask)

    # 获取所有类别 ID
    unique_classes = np.unique(mask)

    # 创建新的 RGB 图像
    rgb_image = Image.new("RGB", mask.shape[::-1])
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    # 填充颜色
    for cls, color in color_map.items():
        colored_mask[mask == cls] = color

    # 将颜色映射应用到图像
    rgb_image = Image.fromarray(colored_mask)

    # 显示图像
    # rgb_image.show()

    # 保存图像（如果提供了输出路径）
    if output_path:
        rgb_image.save(output_path)

# 调用函数进行可视化
# 替换以下路径为你的 PNG 标注文件路径和输出文件路径
labels_dir = 'psv-dataset'
png_file_list = os.listdir(labels_dir)
png_files_path = [os.path.join(labels_dir, png_file) for png_file in png_file_list]

save_dir = 'visualize_test'
os.makedirs(save_dir, exist_ok=True)

color_map = {cls: random_color() for cls in range(6)}

for png_file_path in png_files_path:
    visualize_segmentation(png_file_path, color_map, os.path.join(save_dir,os.path.basename(png_file_path)))
