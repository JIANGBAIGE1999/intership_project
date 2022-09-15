import cv2
import matplotlib.pyplot as plt
import glob

def draw_rect(img, positions, scores=[], labels=[]):
    
    color = (255, 0, 0) # 矩形の色(RGB)。red
    linewidth = 2 # 線の太さ
    font = cv2.FONT_HERSHEY_SIMPLEX #フォント指定
    fontscale = 0.8 #フォントスケール標準が1.0で何倍にするか縮尺
    lineType=cv2.LINE_AA #文字を描画するアルゴリズムの種類
    textsize = 14
    #image = cv2.resize(img,(600,600))
    xresize = img.shape[0]
    yresize = img.shape[1]
    image = img
    for i,pos in enumerate(positions):
        posx = positions[i][:2] #左上頂点座標
        posy = positions[i][2:] #右上頂点座標
        xmin = int(positions[i][0]*xresize)
        ymin = int(positions[i][1]*yresize)
        xmax = int(positions[i][2]*xresize)
        ymax = int(positions[i][3]*yresize)
        #left, top = posx[0], posx[1] #矩形の左上の座標(x, y)をleft, topという変数に格納
        xpos,ypos = [xmin,ymin],[xmax,ymax]
        #print(xpos)
        #print(ypos)
        txpos = (xmax-10, ymax-textsize-linewidth//2) #テキストの描画を開始する座標
        #表示するテキスト日本語と全角記号がNG　半角記号と英数字
        text = " "
        text += str(labels[i]) + ":"
        text += str(scores[i]) + " "
      
        image = cv2.rectangle(image,xpos,ypos,color,linewidth,lineType) #矩形表示
        image = cv2.putText(image,text,txpos,font,fontscale,color,linewidth,lineType) #テキスト表示
    
    return image
