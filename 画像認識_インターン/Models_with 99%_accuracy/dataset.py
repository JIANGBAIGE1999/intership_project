import cv2
import math
import numpy as np
import random

import keras

class DataSequence(keras.utils.Sequence):
    def __init__(self, x, y, param, aug=True):
        """ 初期設定 """
        self.x = x
        self.y = y
        self.indice = np.arange(len(self.x))
        self.aug        = aug
        self.batch_size = param['batch_size']
        self.input_size = param['input_size']
        self.norm       = param['normalization']
        self.gray       = param['gray_scale']
        self.hflip      = param['horizonal_flip']
        self.vflip      = param['vertical_flip']
        self.rotation   = param['rotation']
    
    def __len__(self):
        """ データジェネレータの大きさ """
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        """ バッチ生成 """
        # バッチインデクス取得
        ind = self.indice[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_x = self.x[ind]
        batch_y = self.y[ind]

        # 画像読み込み
        batch_x = np.array([cv2.imread(p)[:,:,::-1] for p in batch_x])
        # 画像のリサイズ処理
        batch_x = np.array([cv2.resize(p, self.input_size) for p in batch_x])

        # データ拡張
        if self.aug:
            batch_x = np.array([self.augment(img) for img in batch_x])

        # グレースケール化
        if self.gray:
            batch_x = np.array([self.rgb2gray(img) for img in batch_x])
            batch_x = batch_x[:, :, :, np.newaxis]

        # 正規化
        if self.norm:
            batch_x = batch_x / 255.0

        return batch_x, batch_y


    def on_epoch_end(self):
        np.random.shuffle(self.indice)

    def augment(self, img):
        """ データ拡張 """
        h, w, _ = img.shape

        #　左右反転
        if self.hflip & np.random.choice([False, True]):
            img = img[:, ::-1, :]

        # 上下反転
        if self.vflip & np.random.choice([False, True]):
            img = img[::-1, :, :]

        # 回転 0 ~ 20[deg]
        if self.rotation:
            center = (h//2, w//2)
            angle = random.uniform(0, 20)
            scale = 1.0
            trans = cv2.getRotationMatrix2D(center, angle, scale)
            img = cv2.warpAffine(img, trans, (h, w))

        return img

    def rgb2gray(self, img):
        """ グレースケール化 """
        return np.array(0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2], dtype='uint8')