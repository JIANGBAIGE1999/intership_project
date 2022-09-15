import os
import cv2
#from buff.defaultbox import get_defaultbox
from defaultbox import get_defaultbox

import matplotlib.pyplot as plt
import glob
import numpy as np

def draw_rect(img, positions):
    positions = np.clip(positions, 0, 1)
    
    color = (255, 0, 0) # 矩形の色(RGB)。red
    linewidth = 2 # 線の太さ
    lineType = cv2.LINE_AA
    xresize = 600
    yresize = 600
    image = img

    for i in range(len(positions)):
        print(positions[i])
        #posx = positions[i][:2] #左上頂点座標
        #posy = positions[i][2:] #右上頂点座標
        xmin = int(positions[i][0]*xresize)
        ymin = int(positions[i][1]*yresize)
        xmax = int(positions[i][2]*xresize)
        ymax = int(positions[i][3]*yresize)
        #exit()

        #left, top = posx[0], posx[1] #矩形の左上の座標(x, y)をleft, topという変数に格納
        xpos,ypos = [xmin,ymin],[xmax,ymax]
        #print(xpos)
        #print(ypos)
        #表示するテキスト日本語と全角記号がNG　半角記号と英数字
        #print(xpos, ypos)
        image = cv2.rectangle(image,xpos,ypos,color,linewidth,lineType) #矩形表示
    #exit()
    return image


img = cv2.imread(os.path.join('RDD2020_data', 'images', 'Adachi_20170906093835.jpg'))

img_size=(40,40,3)

dbox = get_defaultbox(
    image_size=img_size, 
    aspect=[[1,1],[1,2],[1,3]], 
    smin=0.4, 
    smax=1.0, 
    layers=[(5, 5), (3, 3), (1, 1)]
)

result = draw_rect(img, dbox)
cv2.imshow('img', result)
cv2.waitKey(0)
