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
from vis_bbox import draw_rect
import cv2



model = create_model()
# model.summary()

loss = MultiboxLoss(offset=4, classes=2 + 1).multibox_loss

optimizer = 'adam'
model.compile(optimizer=optimizer, loss=loss)

val_gen = Dataset(160, 16, (40, 40, 3), 'val')
model.load_weights(
    filepath='log/test_ssd.hdf5'
)
imgs, anno = val_gen.__getitem__(5)
ori = val_gen.__getorigin__(5)
pred = model.predict(imgs)
mb_loc, mb_conf = pred[0, :, :4], pred[0, :, 4:]
defbox = get_defaultbox(image_size=(40, 40, 3), aspect=[[2,1],[2,3],[2,2]], smin=0.4, smax=1.0, layers=[(5, 5), (3, 3), (1, 1)])
bbox, label, score = decode(defbox, mb_loc, mb_conf)
result = draw_rect(ori, bbox, score, label)

cv2.imwrite('sample.jpg', result)