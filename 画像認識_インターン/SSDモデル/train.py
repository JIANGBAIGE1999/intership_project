import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

from dataset import Dataset
from multi_loss import MultiboxLoss
from model import create_model
from decode import decode
from defaultbox import get_defaultbox


result_path = 'log'
os.makedirs(result_path, exist_ok=True)


model = create_model()
# model.summary()

loss = MultiboxLoss(offset=4, classes=2+1).multibox_loss

optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss=loss)

train_gen = Dataset(640, 16, (40, 40, 3), 'train')
val_gen = Dataset(64, 16, (40, 40, 3), 'val')
history = model.fit(train_gen, 
          epochs=50,
          validation_data=val_gen)
model.save(os.path.join(result_path, 'test_ssd.hdf5'))
plt.figure(figsize=(20, 10))
# Plot training & validation loss values
plt.subplot(121)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
fig_path = os.path.join('log', 'test_log.png')
plt.savefig(fig_path)
#imgs, anno = val_gen.__getitem__(0)
#pred = model.predict(imgs)
#print(pred[0, 0, :])
#model.save(os.path.join(result_path, 'test_ssd.hdf5'))
