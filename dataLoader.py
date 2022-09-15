import numpy as np
import xml.etree.ElementTree as ET
import os
import cv2


def dataLoader(img_size, length, root, ano_path, data_list):
    bbox = []
    imgs = []
    not_bnd = 0
    not_file = 0
    path = None
    for data_path in data_list:
        buf = []
        buf_2 = []
        img = None
        flag = False
        c_width = None
        c_height = None
        if len(imgs) == length:
            break        
        path = os.path.join(root, ano_path, data_path+'.xml')
        try:
            tree = ET.parse(path)
            img = cv2.imread(os.path.join(root, 'images', data_path+'.jpg'))
        except:
            continue
        c_height, c_width  = img.shape[0], img.shape[1]
        chg_h_rate = img_size[0] / c_height
        chg_w_rate = img_size[1] / c_width
        for a in tree.iter():
            if a.tag == 'bndbox':
                flag = True
            if flag:
                if a.tag == 'xmin':
                    buf.append((float(a.text))/c_width)
                if a.tag == 'ymin':
                    buf.append((float(a.text))/c_height)
                if a.tag == 'xmax':
                    buf.append((float(a.text))/c_width)
                if a.tag == 'ymax':
                    buf.append((float(a.text))/c_height)
            if len(buf) == 4:
                buf_2.append(np.array(buf))
                buf = []
                flag = False
        if len(buf_2) == 0:
            continue
        #img = cv2.resize(img, dsize=(img_size[0], img_size[1]))
        imgs.append(img)
        bbox.append(buf_2)
    return imgs, bbox


if __name__ == '__main__':

    """root = 'RDD2020_data'
    ano_path = os.path.join('annotations', 'xmls')
    length = 640
    train_list = None
    train_path = os.path.join(root, 'annotations', 'train.txt')
    with open(train_path, 'r') as f:
        train_list = (f.read()).split('\n')
        if '' in train_list:
            train_list.remove('')
    imgs, bbox = dataLoader(length, root, ano_path, train_list)
    print(len(imgs[0]))
    print(len(bbox[0]))"""
    root = 'RDD2020_data'
    data_list = []
    data_path = os.path.join(root, 'annotations', 'train.txt')
    ano_path = os.path.join('annotations', 'xmls')
    with open(data_path, 'r') as f:
            data_list = (f.read()).split('\n')
            if '' in data_list:
                data_list.remove('')
    img_data, bbox_data =  dataLoader((40,40,3), 10, root, ano_path, data_list)
    print(bbox_data)

