# There are a few options for your data pipeline
# For small dataset, load into numpy array and perform normalization techniques manually

# Or you can use tf/ keras pipelines:
    #tf.keras.preprocessing.image.ImageDataGenerator (Very slow compared to others)
    #tf.keras.preprocessing.image_dataset_from_directory (Recommended. Generates a tf.data.Dataset from image files in a directory.)
    #tf.data.Dataset with image files (Best balance of speed & complexity)
    #tf.data.Dataset with TFRecords (Optimal but complex)
# See: https://towardsdatascience.com/what-is-the-best-input-pipeline-to-train-image-classification-models-with-tf-keras-eb3fe26d3cc5

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

data_dir = 'C:\Users\jgett\OneDrive\Desktop\AI\Projects\Kaggle\Data\dogs'

# Organizing labels
labels = pd.read_csv(data_dir + 'labels.csv')

batch_size = 32
img_height = 180
img_width = 180

# label_mode: String describing the encoding of labels. Options are:
# 'int': means that the labels are encoded as integers (e.g. for sparse_categorical_crossentropy loss).
# 'categorical' means that the labels are encoded as a categorical vector (e.g. for categorical_crossentropy loss).
# 'binary' means that the labels (there can be only 2) are encoded as float32 scalars with values 0 or 1 (e.g. for binary_crossentropy). 
#  None (no labels).

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


