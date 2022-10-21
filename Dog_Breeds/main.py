from models import build_model

from dataLoad import create_dataset

from keras.callbacks import ModelCheckpoint



def train_breeds(DATA_ROOT, BATCH_SIZE, EPOCHS):
    
    train_ds, val_ds = create_dataset(DATA_ROOT, BATCH_SIZE)
    
    model = build_model
    
    checkpoint = ModelCheckpoint("dog_breeds.hdf5",monitor='val_loss',verbose=1,mode='min',save_best_only=True,save_weights_only=True)

    model.fit(train_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint],
        validation_data=val_ds)
    
def test_breeds()
    
