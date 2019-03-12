#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/3/11
"""
import os
from PIL import Image

from prep.dataset_prep import list_dataset
from root_dir import DATASET_DIR, DATA_DIR
from utils.project_utils import *


def draw_multi_imgs(path_list, file_name):
    """
    绘制相似图片组
    :param path_list: 图片路径列表
    :param file_name: 输出文件名
    :return: None
    """
    img_w, img_h = 4, 3
    img_size = 416

    try:
        o_images = [Image.open(p) for p in path_list]
        images = []

        for img in o_images:
            wp = img_size / float(img.size[0])
            hsize = int(float(img.size[1]) * float(wp))
            img = img.resize((img_size, hsize), Image.ANTIALIAS)
            images.append(img)

    except Exception as e:
        print('Exception: {}'.format(e))
        return

    new_im = Image.new('RGB', (img_size * img_w, img_size * img_h), color=(255, 255, 255))

    x_offset, y_offset = 0, 0
    for i in range(img_h):
        for j in range(img_w):
            im = images[i * img_w + j]
            new_im.paste(im, (x_offset, y_offset))
            x_offset += 416
        y_offset += 416
        x_offset = 0

    new_im.save(file_name)  # 保存图片


def main():
    new_train = os.path.join(DATASET_DIR, 'train')
    new_test = os.path.join(DATASET_DIR, 'test')

    cats_dict, dogs_dict = list_dataset(new_train)
    cats_list = list(cats_dict.values())
    dogs_list = list(dogs_dict.values())
    random.shuffle(cats_list)
    random.shuffle(dogs_list)

    draw_multi_imgs(cats_list[:12], os.path.join(DATA_DIR, 'train_cat.jpg'))
    draw_multi_imgs(dogs_list[:12], os.path.join(DATA_DIR, 'train_dog.jpg'))


if __name__ == '__main__':
    main()
