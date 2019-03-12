#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/3/12
"""
import os
import sys
import tensorflow as tf

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from root_dir import DATASET_DIR


def main():
    image_height, image_width = 150, 150
    train_dir = os.path.join(DATASET_DIR, 'train')
    print(train_dir)
    test_dir = os.path.join(DATASET_DIR, 'test')

    no_classes = 2
    no_validation = 800

    epochs = 2
    batch_size = 200

    no_train = 2000
    no_test = 800

    input_shape = (image_height, image_width, 3)
    epoch_steps = no_train // batch_size
    test_steps = no_test // batch_size

    simple_cnn_model = simple_cnn(input_shape, no_classes)

    generator_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    generator_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    train_images = generator_train.flow_from_directory(
        train_dir,
        batch_size=batch_size,
        target_size=(image_width, image_height)
    )

    test_images = generator_test.flow_from_directory(
        test_dir,
        batch_size=batch_size,
        target_size=(image_width, image_height)
    )

    simple_cnn_model.fit_generator(
        train_images,
        steps_per_epoch=epoch_steps,
        epochs=epochs,
        validation_data=test_images,
        validation_steps=test_steps
    )


def simple_cnn(input_shape, no_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=input_shape
    ))
    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(units=no_classes, activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    main()
