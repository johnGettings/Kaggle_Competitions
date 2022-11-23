import pandas as pd
from sklearn.preprocessing import LabelEncoder

from models import build_model
from dataLoad import create_dataset

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.resnet50 import preprocess_input


def train_breeds(DATA_ROOT, BATCH_SIZE, EPOCHS):
    
    train_ds, val_ds = create_dataset(DATA_ROOT, BATCH_SIZE)
    
    model = build_model
    
    checkpoint = ModelCheckpoint("dog_breeds.hdf5",monitor='val_loss',verbose=1,mode='min',save_best_only=True,save_weights_only=True)

    model().fit(train_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint],
        validation_data=val_ds)
    
def test_breeds(WEIGHTS, IMAGE_PATH, LABELS_PATH):
    
    #Preprocessing Image
    def process_img(filename):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = preprocess_input(img)
        return img

    img_paths = []
    img_paths.append(IMAGE_PATH)
    ds_test = tf.data.Dataset.from_tensor_slices((img_paths))
    ds_test = ds_test.map(process_img).batch(1)

    #Building model
    model = build_model(model_weights=WEIGHTS)

    #Guessing label
    labels = pd.read_csv(LABELS_PATH)
    le = LabelEncoder()
    int_input = le.fit_transform(labels.iloc[:,1].values)
    labels['int_class'] = int_input
    
    #Prediction
    confidences = model.predict(ds_test)
    guess_loc = tf.argmax(confidences, axis=-1).numpy()[0]
    
    print(labels[labels['int_class']==guess_loc].iloc[0]['breed'])
    print("confidence: " + str(confidences[0,guess_loc]))