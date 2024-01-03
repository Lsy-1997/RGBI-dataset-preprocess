import json
import numpy as np
from PIL import Image
import random
import os
from tqdm import tqdm

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def decode_rle(counts, size):
    rle = np.array(counts)
    rle_pairs = rle.reshape(-1, 2)
    img = np.zeros(size[0] * size[1], dtype=np.uint8)
    pos = 0
    for count, val in rle_pairs:
        img[pos:pos+count] = val
        pos += count
    return img.reshape(size)

def rle2mask(mask_rle, shape):
    odd, even = [np.asarray(x, dtype=int)
                       for x in (mask_rle[::2], mask_rle[1::2])]
    assert len(odd) >= len(even)

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    start = 0
    for i in range(len(even)):
        start += odd[i]
        end = start + even[i]
        img[start: end] = 255
        start += even[i]

    return img.reshape(shape)

# 解析 JSON 数据
# json_path = 'dataset/annotations.json'
# json_path = 'dataset/吴思宇标注20231227编号尾号(0-3)annotations.json'

# json_path = 'dataset/吴思宇annotations_lsy精标all.json'
json_path = 'dataset/annotations_lsy精标.json'

# 打开并读取 JSON 文件
with open(json_path, 'r') as file:
    json_data = json.load(file)

# 为每个 category_id 分配颜色，设0为背景类，类别从1开始起算
label_color_file = 'label_color.json'
if not os.path.exists(label_color_file):
    category_colors = {cat['id'] + 1: random_color() for cat in json_data['categories']}
    with open('label_color.json', 'w') as json_file:
        json.dump(category_colors, json_file, indent=4)
else:
    with open(label_color_file, 'r') as file:
        category_colors = json.load(file)
        category_colors = {int(k): tuple(v) for k, v in category_colors.items()}

png_labels_save_dir = 'dataset/labels'
os.makedirs(png_labels_save_dir, exist_ok=True)

visualized_labels_save_dir = 'dataset/visualized_labels'
os.makedirs(visualized_labels_save_dir, exist_ok=True)

skipped_num = 0
# 处理每个图像
for image_info in tqdm(json_data['images']):
    image_id = image_info['id']
    image_size = (image_info['height'], image_info['width'])
    mask = np.zeros(image_size, dtype=np.uint8)
    file_name = os.path.splitext(os.path.basename(image_info['file_name']))[0]

    # 处理与此图像相关的所有标注
    for annotation in json_data['annotations']:
        if annotation['image_id'] == image_id:
            # 设0为背景类，类别从1开始起算
            category_id = annotation['category_id'] + 1
            color = category_colors[category_id]
            rle_counts = annotation['segmentation']['counts']
            # annotation_mask = decode_rle(rle_counts, image_size)
            annotation_mask = rle2mask(rle_counts, image_size)
            mask[annotation_mask == 255] = category_id

    # 若为空标注则不保存
    if np.all(mask==0):
        skipped_num+=1
        continue
    # 保存掩膜为png格式
    mask_image = Image.fromarray(mask)
    mask_image.save(os.path.join(png_labels_save_dir, file_name + '.png'))

    # 将掩膜转换为彩色图像
    color_mask = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    for value, color in category_colors.items():
        color_mask[mask == value] = color

    # 保存彩色图像
    color_mask_image = Image.fromarray(color_mask)
    color_mask_image.save(os.path.join(visualized_labels_save_dir, file_name + '.png'))

print(f"skipped {skipped_num} void labels")
print('finished')
