import os
from os.path import join # operating system file paths
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting graphs

import tensorflow as tf
from tensorflow import keras
from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, MaxPool2D, Dropout, Flatten, AveragePooling2D
from keras.applications.resnet import ResNet50


def build_model(model_weights=None):
    # Augmentation Layers
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ]
    )
    
    # input layers
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)

    # transfer learning
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))(x)

    # creating our new model head to combine with the ResNet base model
    head_model = MaxPool2D(pool_size=(4, 4))(base_model)
    head_model = Flatten(name='flatten')(head_model)
    head_model = Dense(1024, activation='relu')(head_model)
    head_model = Dropout(0.2)(head_model)
    head_model = Dense(512, activation='relu')(head_model)
    head_model = Dropout(0.2)(head_model)
    head_model = Dense(120, activation='softmax')(head_model)

    # final configuration
    model = Model(inputs, head_model)

    # RESNET not trainable
    model.layers[2].trainable = False

    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    
    if model_weights is not None:
        model.load_weights(model_weights)
            
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model