#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/3/12
"""
import os
import sys

import keras
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import DATASET_DIR


def main():
    """
    loss: 0.4562 - acc: 0.7755 - val_loss: 0.4220 - val_acc: 0.7950
    augmentation:
    """
    image_height, image_width = 150, 150
    train_dir = os.path.join(DATASET_DIR, 'train')
    test_dir = os.path.join(DATASET_DIR, 'test')

    no_classes = 2  # 猫狗两类
    no_validation = 800

    epochs = 10
    batch_size = 20

    no_train = 2000
    no_test = 800

    input_shape = (image_height, image_width, 3)
    epoch_steps = no_train // batch_size
    test_steps = no_test // batch_size

    simple_cnn_model = simple_cnn(input_shape, no_classes)

    generator_train = ImageDataGenerator(
        rescale=1. / 255.,
        horizontal_flip=True,
        zoom_range=0.3,
        shear_range=0.3
    )
    generator_test = ImageDataGenerator(rescale=1. / 255.)

    train_generator = generator_train.flow_from_directory(
        train_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size
    )

    # 查看数据格式
    # for data_batch, labels_batch in train_generator:
    #     print(data_batch.shape)
    #     print(labels_batch.shape)
    #     print(labels_batch)  # label是oh形式
    #     break

    test_generator = generator_test.flow_from_directory(
        test_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size
    )

    simple_cnn_model.fit_generator(
        train_generator,
        steps_per_epoch=epoch_steps,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=test_steps
    )


def simple_cnn(input_shape, no_classes):
    """
    网络输入的尺寸，类别数
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(no_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    main()
