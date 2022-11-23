import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.resnet50 import preprocess_input


def load(DATA_ROOT):
    train_dir = DATA_ROOT + 'train/'

    # Reading csv
    labels = pd.read_csv(DATA_ROOT + 'labels.csv')
    print(labels.head) 

    # Creating array of image paths
    img_paths = []
    for id in labels['id']:
        img_paths.append(train_dir + id + ".jpg")
        
    # Creating integer classes
    le = LabelEncoder()
    int_input = le.fit_transform(labels.iloc[:,1].values)
    labels['int_class'] = int_input
    
    # Creating one hot encoded labels
    cat_count = 120  #depth or the number of categories
    oh_input = tf.one_hot(int_input, cat_count) #apply one-hot encoding
    
    return(oh_input, img_paths)

def read_and_decode(filename, label):
    # Returns a tensor with byte values of the entire contents of the input filename.
    img = tf.io.read_file(filename)
    # Decoding raw JPEG tensor data into 3D (RGB) uint8 pixel value tensor
    img = tf.io.decode_jpeg(img, channels=3)
    #Resize
    IMG_SIZE=[224,224]
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    return img, label

def ds_split(ds, ds_size, shuffle_size, train_split=0.8, val_split=0.2, shuffle=True):
    assert (train_split + val_split) == 1
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=99)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    
    return train_ds, val_ds

def create_dataset(DATA_ROOT, BATCH_SIZE):
    oh_input, img_paths = load(DATA_ROOT)
    ds_oh = tf.data.Dataset.from_tensor_slices((img_paths, oh_input))
    ds_oh = ds_oh.map(read_and_decode)
    train_ds, val_ds = ds_split(ds_oh, len(img_paths), len(img_paths), train_split=0.8, val_split=0.2, shuffle=True)
    
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(buffer_size=len(img_paths), reshuffle_each_iteration=True)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.cache()
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds