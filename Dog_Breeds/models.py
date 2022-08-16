import os
from os.path import join # operating system file paths
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting graphs

import tensorflow as tf
from keras.utils.image_utils import load_img, img_to_array # loading image to PIL, then PIL to numpy
from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, SGD, Adam
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout, Flatten, AveragePooling2D
from keras.utils import np_utils, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input

# taking the popular ResNet model trained on imagenet, and excluding the top of the network.
# include_top=False exludes the last global pooling and fully connected prediciton layer
# so final layer is of size (7, 7, 2048)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# base_model.summary() # to view NN layers

# freezing bottom layers so weights are not adjusted during training
# the feature extraction (conv, pooling) sections stay intact 
for layer in base_model.layers:
    layer.trainable = False

# creating our new model head to combine with the ResNet base model
head_model = MaxPool2D(pool_size=(4, 4))(base_model.output)
head_model = Flatten(name='flatten')(head_model)
head_model = Dense(1024, activation='relu')(head_model)
head_model = Dropout(0.3)(head_model)
head_model = Dense(512, activation='relu')(head_model)
head_model = Dropout(0.3)(head_model)
head_model = Dense(120, activation='softmax')(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

# Categorical crossentropy loss should be used when determining labels from many classes
#    and only if the output values are one hot encoded
# Adam optimizer should be your default choice as a general purpose optimizer 
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
