#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/3/11
"""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 项目根目录

DATA_DIR = os.path.join(ROOT_DIR, 'data')  # 数据
DATASET_DIR = os.path.join(ROOT_DIR, '..', 'data_set/dogs-vs-cats-new')  # 数据集

O_DATASET_DIR = os.path.join(ROOT_DIR, '..', 'data_set/dogs-vs-cats/train')  # 数据集
