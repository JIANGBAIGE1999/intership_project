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

loss = MultiboxLoss(offset=4, classes=2 + 1).multibox_loss

optimizer = 'adam'
model.compile(optimizer=optimizer, loss=loss)

train_gen = Dataset(640, 16, (40, 40, 3))
val_gen = Dataset(160, 16, (40, 40, 3))
print(type(train_gen.__getitem__(0)[0]))