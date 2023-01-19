import numpy as np
import random
import cv2
from defaultbox import get_defaultbox
import os
from dataLoader import dataLoader

# 正解ボックスとデフォルトボックスとのIoUの計算
def compute_iou(dbox, bbox):
    # かぶっている領域の計算
    inter_upleft = np.maximum(dbox[:, :2], bbox[:2])
    inter_botright = np.minimum(dbox[:, 2:], bbox[2:])
    inter_wh = inter_botright - inter_upleft
    inter_wh = np.maximum(inter_wh+1, 0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    # 正解ボックスとデフォルトボックスの面積の計算
    area_pred = np.maximum((bbox[2] - bbox[0])+1, 0) * np.maximum((bbox[3] - bbox[1])+1, 0)
    area_gt = np.maximum((dbox[:, 2] - dbox[:, 0])+1, 0)
    area_gt *= np.maximum((dbox[:, 3] - dbox[:, 1])+1, 0)
    # IoUの計算
    union = area_pred + area_gt - inter
    iou = inter/union
    return iou


def encode(defaultbox, bbox, label, iou_thresh=0.3, variance=[0.3, 0.5]):
    """ ここを作成 """
    conf = []
    loc = []
    idx = 0
    conf_pre = []
    loc_pre = []
    iou_log = []
    esp = 1e-5
    #print(label.shape)
    for box in bbox:
        conf_buf = np.zeros((defaultbox.shape[0], label.shape[1]))
        encoded_box = np.zeros((defaultbox.shape))
        #print(conf_buf.shape)
        iou = compute_iou(defaultbox, box)
        iou_log.append(iou)
        iou_mask = iou > iou_thresh # iouの閾値を超えているかのmaskを作成
        if not iou_mask.any():
            iou_mask[iou.argmax()] = True
        assigned_box = defaultbox # iouの閾値を超えたデフォルトボックスを取り出し
        #assigned_box = defaultbox[iou_mask]
        # オフセットを求める
        bbox_center = 0.5 * (box[:2]+box[2:])
        bbox_wh = box[2:] - box[:2]
        assigned_box_center = 0.5 * (assigned_box[:, :2] + assigned_box[:, 2:])
        assigned_box_wh =(assigned_box[:, 2:4] - assigned_box[:, :2])
        """encoded_box[:, :, :2][iou_mask] = bbox_center - assigned_box_center
        encoded_box[:, :, :2][iou_mask] /= assigned_box_wh
        encoded_box[:, :, :2][iou_mask] /= variance[0]
        encoded_box[:, :, 2:][iou_mask] = np.log(bbox_wh / assigned_box_wh)
        encoded_box[:, :, 2:][iou_mask] /= variance[1]"""
        """encoded_box[:, :2][iou_mask] = bbox_center - assigned_box_center
        encoded_box[:, :2][iou_mask] /= assigned_box_wh
        encoded_box[:, :2][iou_mask] /= variance[0]
        encoded_box[:, 2:][iou_mask] = np.log(bbox_wh / assigned_box_wh)
        encoded_box[:, 2:][iou_mask] /= variance[1]"""
        encoded_box[:, :2] = bbox_center - assigned_box_center[:, :]
        encoded_box[:, :2] /= (assigned_box_wh[:, :]*variance[0])
        encoded_box[:, 2:] = (np.log((bbox_wh / assigned_box_wh[:, :])+esp)/variance[1])
        loc_pre.append(encoded_box)
        # 使用するデフォルトボックスについての正解ラベルの取り出し
        #print(label[idx, :])
        conf_buf[:, 0][iou_mask] = label[idx,:]
        conf_pre.append(conf_buf)
        idx += 1
    bbox_len = len(bbox)
    for i in range((iou_log[0]).shape[0]):
        max_idx = 0
        max_iou = iou_log[0][i]
        for l in range(bbox_len-1):
            if max_idx < iou_log[l+1][i]:
                max_idx = l+1
                max_iou = iou_log[l+1][i]
        loc.append(loc_pre[max_idx][i])
        conf.append(conf_pre[max_idx][i])
    """for i in range(len(iou_log)):
        count = 0
        for idx in range(iou_log[i].shape[0]):
            if iou_log[i][idx] > 0.5:
                count += 1
        if count >= 5:
            print(i)
            print(count)"""
    return np.array(loc), np.array(conf)


if __name__ == '__main__':
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
        d_list = np.array([draw(zmap, pt1, pt2, random.choice([1, 1])) for pt1, pt2 in zip(left_upper, right_bottom)])
        return zmap, np.array(d_list)
    """dbox = np.array([
            [
                [0.1, 0.1, 1.1, 1.1],
                [0.5, 0.5, 1.5, 1.5],
                [0.2, 0.2, 2.2, 2.2],
                [1.0, 1.0, 3.0, 3.0]
            ],
            [
                [0.1, 0.1, 1.1, 1.1],
                [0.5, 0.5, 1.5, 1.5],
                [0.2, 0.2, 2.2, 2.2],
                [1.0, 1.0, 3.0, 3.0]
            ],
        ])"""
    #print(dbox.shape)
    """ ここにテストコードを作成 """
    dbox = get_defaultbox(image_size=(40,40,3), 
                                aspect=[[1,1],[1,2],[1,3]], 
                                smin=0.4, 
                                smax=1.0, 
                                layers=[(5, 5), (3, 3), (1, 1)])
    #print(dbox.shape)
    """imgs = []
    anno = []
    for i in range(4):
        img, ano = gen_sample((40,40,3))
        img = img / 255.0
        bbox = ano[:, :4].astype(np.float32)
        label = (np.array([[1], [1], [1], [1]])).astype(np.int32)
        loc, conf = encode(dbox, bbox, label)
        loc_conf = np.concatenate([loc, conf], axis=-1)
        imgs.append(img)
        anno.append(loc_conf)
    imgs = np.array(imgs)
    anno = np.array(anno)"""
    #print(anno)

    imgs = []
    anno = []
    root = 'RDD2020_data'
    ano_path = os.path.join('annotations', 'xmls')
    data_list = None
    data_path = os.path.join(root, 'annotations', 'train.txt')
    with open(data_path, 'r') as f:
        data_list = (f.read()).split('\n')
        if '' in data_list:
            data_list.remove('')
    img_data, bbox_data =  dataLoader((40, 40, 3), 5, root, ano_path, data_list)

    for i in range(5):
        buf = []
        img = img_data[i]
        img = img / 255.0
        buf = []
        bbox = bbox_data[i]
        label = np.ones((len(bbox), 1))
        loc, conf = encode(dbox, bbox, label)
        loc_conf = np.concatenate([loc, conf], axis=-1)
        imgs.append(img)
        anno.append(loc_conf)
    anno = np.array(anno)
