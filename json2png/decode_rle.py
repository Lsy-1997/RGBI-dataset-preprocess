import json
import cv2
import numpy as np


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



with open('dataset/annotations.json') as f:
    data = json.load(f)

# data['images']

for anno in data['annotations']:
    img_size = anno['segmentation']['size']
    mask = rle2mask(anno['segmentation']['counts'], img_size)

    cv2.imshow('tmp', mask)
    cv2.waitKey(0)
