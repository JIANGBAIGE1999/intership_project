import cv2
import random
import numpy as np
import math

import tensorflow as tf
import keras

from encode import encode
from defaultbox import get_defaultbox

from dataLoader import dataLoader
import os


class Dataset(tf.keras.utils.Sequence):
    def __init__(self, length, batch, img_size, phase, aug=False):
        self.length = length
        self.batch = batch
        self.img_size = img_size
        self.aug = aug
        self.phase = phase
        self.defaultbox = get_defaultbox(image_size=img_size, 
                                         aspect=[[1,1],[1,2],[1,3]], 
                                         smin=0.4, 
                                         smax=1.0, 
                                         layers=[(5, 5), (3, 3), (1, 1)])
        self.origin_img = []

    def __len__(self):
        return math.ceil(self.length / self.batch)

    def __getitem__(self, index):
        imgs = []
        anno = []
        root = 'D:\BaiduNetdiskDownload\RDD2020_data\RDD2020_data'
        ano_path = os.path.join('annotations', 'xmls')
        data_list = None
        if self.phase == 'train':
            data_path = os.path.join(root, 'annotations', 'train.txt')
        elif self.phase == 'val':
            data_path = os.path.join(root, 'annotations', 'val.txt')
        with open(data_path, 'r') as f:
            data_list = (f.read()).split('\n')
            if '' in data_list:
                data_list.remove('')
        img_data, bbox_data =  dataLoader(self.img_size, self.length, root, ano_path, data_list)
        for i in range(index*self.batch, (index+1)*self.batch):
            buf = []
            img = img_data[i]
            self.origin_img.append(img)
            img = cv2.resize(img, dsize=(self.img_size[0], self.img_size[1]))
            img = img / 255.0
            buf = []
            bbox = bbox_data[i]
            label = np.ones((len(bbox), 1))
            loc, conf = encode(self.defaultbox, bbox, label)
            loc_conf = np.concatenate([loc, conf], axis=-1)
            imgs.append(img)
            anno.append(loc_conf)
        return np.array(imgs), np.array(anno)
    
    def __getorigin__(self,index):
        return self.origin_img[index]

def make_circle(zmap, c, r):
    xmin = c[0] - r
    ymin = c[1] - r
    xmax = c[0] + r
    ymax = c[1] + r
    zmap = cv2.circle(zmap, tuple(c), r, (255, 0, 0), -1)
    # zmap = cv2.rectangle(zmap,(xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    ann = [ymin, xmin, ymax, xmax, 0]
    return ann


def make_square(zmap, c, r):
    xmin = c[0] - r
    ymin = c[1] - r
    xmax = c[0] + r
    ymax = c[1] + r
    zmap = cv2.rectangle(zmap,(xmin, ymin), (xmax, ymax), (0, 255, 0), -1)
    # zmap = cv2.rectangle(zmap,(xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    ann = [ymin, xmin, ymax, xmax, 1]
    return ann


def draw(img, p1, p2, circle=True):
    w, h = p2 - p1
    r = np.random.randint(w // 4, w // 2)
    # x = np.random.randint(w // 4, 3 * w // 4) + p1[0]
    # y = np.random.randint(h // 4, 3 * h // 4) + p1[1]
    x = p1[0] + w // 2
    y = p1[1] + h // 2
    if circle:
        return make_circle(img, (x, y), r)
    else:
        return make_square(img, (x, y), r)


def gen_sample(map_size=(32, 32, 3)):
    zmap = np.zeros(map_size)
    w, h = map_size[0], map_size[1]
    left_upper = np.array([[0, 0], [w // 2, 0], [0, h // 2], [w // 2, h // 2]])
    right_bottom = np.array([[w // 2, h // 2], [w, h // 2], [w // 2, h], [w, h]])
    d_list = np.array([draw(zmap, pt1, pt2, random.choice([True, False])) for pt1, pt2 in zip(left_upper, right_bottom)])
    return zmap, np.array(d_list)
