#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/3/11
"""

from root_dir import O_DATASET_DIR, DATASET_DIR
from utils.project_utils import *


def list_dataset(dataset_dir):
    """
    将训练数据集读入内存，分为猫和狗两部分
    """
    paths_list, names_list = traverse_dir_files(dataset_dir)
    cats_dict, dogs_dict = dict(), dict()

    for path, name in zip(paths_list, names_list):
        [clz, num, _] = name.split('.')
        num = int(num)
        if clz == 'cat':
            cats_dict[num] = path
        elif clz == 'dog':
            dogs_dict[num] = path
        else:
            continue

    # print('cat: {}, dog: {}'.format(len(cats_dict.keys()), len(dogs_dict.keys())))
    return cats_dict, dogs_dict


def copy_files(target_folder, clz_name, n_start, n_end):
    cats_dict, dogs_dict = list_dataset(O_DATASET_DIR)

    new_train = os.path.join(DATASET_DIR, 'train')
    new_test = os.path.join(DATASET_DIR, 'test')

    mkdir_if_not_exist(DATASET_DIR)
    mkdir_if_not_exist(new_train)
    mkdir_if_not_exist(new_test)

    # 测试数据
    # target_folder = 'train'
    # clz_name = 'cat'
    # n_start = 0
    # n_end = 10

    for i in range(n_start, n_end):
        data_dict = cats_dict if clz_name == 'cat' else dogs_dict
        folder = new_train if target_folder == 'train' else new_test
        folder = os.path.join(folder, clz_name)
        mkdir_if_not_exist(folder)
        shutil.copy(data_dict[i], folder)
    print("[完成]目标文件夹: {}, 类别: {}, 起止: {} ~ {}".format(
        target_folder, clz_name, n_start, n_end))


def main():
    # 1000张猫+1000张狗训练集；400张猫+400张狗测试集
    copy_files('train', 'cat', 0, 1000)
    copy_files('train', 'dog', 0, 1000)
    copy_files('test', 'cat', 0, 400)
    copy_files('test', 'dog', 0, 400)


if __name__ == '__main__':
    main()
