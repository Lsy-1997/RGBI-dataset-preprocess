import os
import shutil
import random

def split_dataset(base_dir, save_dir, val_ratio):
    # 文件夹路径
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')
    rgbi_images_dir = os.path.join(base_dir, 'images_rgbi')

    # 创建训练和验证集文件夹
    for folder in ['images', 'labels', 'images_rgbi']:
        for subdir in ['train', 'val']:
            os.makedirs(os.path.join(save_dir, folder, subdir), exist_ok=True)

    # 获取所有图像文件名
    all_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    # 打乱并分配训练集和验证集
    random.shuffle(all_images)
    num_val = int(len(all_images) * val_ratio)
    val_images = all_images[:num_val]
    train_images = all_images[num_val:]

    # 复制文件到相应文件夹
    for image_name in train_images:
        # 拷贝images
        shutil.copy(os.path.join(images_dir, image_name), os.path.join(save_dir, 'images', 'train', image_name))
        # 拷贝images_rgbi
        rgbi_image_name = os.path.splitext(image_name)[0] + '.png'
        shutil.copy(os.path.join(rgbi_images_dir, rgbi_image_name), os.path.join(save_dir, 'images_rgbi', 'train', rgbi_image_name))
        # 拷贝labels
        label_name = os.path.splitext(image_name)[0] + '.png'
        shutil.copy(os.path.join(labels_dir, label_name), os.path.join(save_dir, 'labels', 'train', label_name))

    for image_name in val_images:
        shutil.copy(os.path.join(images_dir, image_name), os.path.join(save_dir, 'images', 'val', image_name))
        rgbi_image_name = os.path.splitext(image_name)[0] + '.png'
        shutil.copy(os.path.join(rgbi_images_dir, rgbi_image_name), os.path.join(save_dir, 'images_rgbi', 'val', rgbi_image_name))
        label_name = os.path.splitext(image_name)[0] + '.png'
        shutil.copy(os.path.join(labels_dir, label_name), os.path.join(save_dir, 'labels', 'val', label_name))

# 使用示例
base_directory = 'dataset'  # 替换为您的数据集根目录路径
save_directory = 'splitted_dataset'
validation_ratio = 0.2  # 20% 的数据作为验证集
split_dataset(base_directory, save_directory, validation_ratio)
