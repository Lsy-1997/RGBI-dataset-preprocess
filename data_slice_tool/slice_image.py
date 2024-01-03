from PIL import Image
import os
import numpy as np
import json
from tqdm import tqdm

label_color_file = 'label_color.json'
with open(label_color_file, 'r') as file:
    category_colors = json.load(file)
    category_colors = {int(k): tuple(v) for k, v in category_colors.items()}

def slice_image(image_path, label_path, rgbi_path, output_folder, slice_num):
    
    # 读取图像和标注文件
    img = Image.open(image_path)
    ann = Image.open(label_path)
    rgbi = Image.open(rgbi_path)

    file_name = os.path.splitext(os.path.basename(image_path))[0]

    # 计算每个正方形切片的尺寸
    width, height = img.size
    slice_size = height  # 使用高度作为正方形的边长

    step_size = (width - slice_size) // (slice_num - 1)

    # 切割并保存图像
    for i in range(slice_num):
        left = i * step_size
        right = left + slice_size
        top = 0
        bottom = slice_size

        # 切割图像和标注文件
        img_slice = img.crop((left, top, right, bottom))
        ann_slice = ann.crop((left, top, right, bottom))
        rgbi_slice = rgbi.crop((left, top, right, bottom))

        # 将掩膜转换为彩色图像
        mask = np.array(ann_slice)
        color_mask = np.zeros((slice_size, slice_size, 3), dtype=np.uint8)
        for value, color in category_colors.items():
            color_mask[mask == value] = color

        color_mask_image = Image.fromarray(color_mask)

        # 保存切割后的图像和标注
        if os.path.basename(os.path.dirname(image_path)) == 'train':
            img_output_dir = os.path.join(output_folder, 'images','train')
            label_output_dir = os.path.join(output_folder, 'labels','train')
            rgbi_output_dir = os.path.join(output_folder, 'images_rgbi','train')
            rgb_label_output_dir = os.path.join(output_folder, 'visualized_labels','train')
        elif os.path.basename(os.path.dirname(image_path)) == 'val':
            img_output_dir = os.path.join(output_folder, 'images','val')
            label_output_dir = os.path.join(output_folder, 'labels','val')
            rgbi_output_dir = os.path.join(output_folder, 'images_rgbi','val')
            rgb_label_output_dir = os.path.join(output_folder, 'visualized_labels','val')
        os.makedirs(img_output_dir,exist_ok=True)
        os.makedirs(label_output_dir,exist_ok=True)
        os.makedirs(rgbi_output_dir,exist_ok=True)
        os.makedirs(rgb_label_output_dir,exist_ok=True)
        img_slice.save(f'{img_output_dir}/{file_name}_slice_{i}.jpg')
        ann_slice.save(f'{label_output_dir}/{file_name}_slice_{i}.png')
        rgbi_slice.save(f'{rgbi_output_dir}/{file_name}_slice_{i}.png')
        color_mask_image.save(f'{rgb_label_output_dir}/{file_name}_slice_{i}.png')

# 调用函数进行切片
# 替换以下路径为你的原始图像文件路径和标注文件路径，以及输出文件夹路径

for dir in ['train', 'val']:
    imgs_dir = f'splitted_dataset/images/{dir}'
    labels_dir = f'splitted_dataset/labels/{dir}'
    rgbi_dir = f'splitted_dataset/images_rgbi/{dir}'
    output_dir = f'splitted_dataset_slice/'

    print(f'processing {dir} set')

    imgs_file = [f for f in os.listdir(imgs_dir) if f.endswith('.jpg')]
    imgs_path = [os.path.join(imgs_dir, img) for img in imgs_file]

    rgbi_path = [os.path.join(rgbi_dir, os.path.splitext(file)[0]+'.png') for file in imgs_file]

    labels_path = [os.path.join(labels_dir, os.path.splitext(file)[0]+'.png') for file in imgs_file]

    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(len(imgs_path))):
        slice_image(imgs_path[i], labels_path[i], rgbi_path[i], output_dir, 4)
