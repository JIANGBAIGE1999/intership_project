#!/usr/bin/env python
# coding: utf-8

# In[42]:


import itertools
import math
import numpy as np

# In[43]:


def storage(cx,cy,w,h,result):    
    x_min = cx - w/2
    y_min = cy - h/2
    x_max = cx + w/2
    y_max = cy + h/2
    result.append([x_min, y_min, x_max, y_max])
    return result

def get_defaultbox(image_size=(40, 40, 3), aspect=[], smin=0.4, smax=1.0, layers=[(5, 5), (3, 3), (1, 1)]): 
    result = []
    feature_maps = []#特徴量マップのサイズを集めること
    s_k = []
    Aspect = []#アスペクト比の値を保存する
    steps = []
    scales_min = []
    scales_max = []
    #result_c = []


    for layer in layers:
        s = image_size[0]/layer[0]
        txt = '{}'.format(s)
        _, txt = txt.split('.')
        if int(txt[0]) >= 5:
            s += 1
        steps.append(math.floor(s))
    
    if len(layers) == 1:
        s_k.append(1)
    else:#スケールを計算すること
        for i in range(len(layers)):
            s_k.append(smin + ((smax-smin)*i/(len(layers)-1)))
    
    for sk in s_k:
        scales_min.append(image_size[0] * sk)
    dummy = scales_min[1] - scales_min[0]
    for i in range(1, len(scales_min)):
        if i == (len(scales_min)-1):
            scales_max.append(scales_min[i])
            scales_max.append(scales_min[i] + dummy)
        else:
            scales_max.append(scales_min[i])

    if aspect == []:
        aspect = [2,3,1]
        Aspect = aspect

    else: 
        #アスペクト比を計算すること
        for child in aspect:
            Aspect.append(child[0]/child[1])

    for length in layers:
        feature_maps.append(length[0])
    
    Aspect.remove(1)
        
    #特徴量マップのサイズを抽出すること
    for k, f in enumerate(feature_maps):
        for i, j in itertools.product(range(f), repeat=2):
            f_k = image_size[0] / steps[k]
            #fk = image_size[0]/f
            # デフォルトボックスの中央座標 x,yを計算し、具体的には論文にある
            cx = (j + 0.5) / f_k
            cy = (i + 0.5) / f_k
            sk = scales_min[k]/image_size[0]
            result = storage(cx, cy, sk, sk, result)
            sk_prime = math.sqrt(sk * (scales_max[k] / image_size[0]))
            result = storage(cx, cy, sk_prime, sk_prime, result)

            for temp in Aspect:
                """if temp == 1 and k != len(feature_maps)-1:
                    weight = math.sqrt(s_k[k]*s_k[k+1])
                    height = math.sqrt(s_k[k]*s_k[k+1])
                    #アスペクト比の特殊の[x_min,y_min,x_max,y_max]を追加すること
                    result = storage(cx,cy,weight,height,result)
                    weight = s_k[k]
                    height = s_k[k]
                    result = storage(cx,cy,weight,height,result)
                elif temp == 1 and k == len(feature_maps)-1:
                    weight = math.sqrt(s_k[k]*1.05)
                    height = math.sqrt(s_k[k]*1.05)
                    result = storage(cx,cy,weight,height,result)
                    weight = s_k[k]
                    height = s_k[k]
                    result = storage(cx,cy,weight,height,result)"""
                #else:
                weight = sk*math.sqrt(temp)
                height = sk/math.sqrt(temp)
                #一般的の[x_min,y_min,x_max,y_max]を追加すること
                #cx, cy, weight, height = np.clip(np.array([cx, cy, weight, height]), 0, 1)
                result = storage(cx,cy,weight,height,result)

                weight = sk/math.sqrt(temp)
                height = sk*math.sqrt(temp)
                #cx, cy, weight, height = np.clip(np.array([cx, cy, weight, height]), 0, 1)
                result = storage(cx,cy,weight,height,result)
            #result_c.append([cx*600, cy*600])
    result = np.array(result)
    result = np.clip(result, 0, 1)
    return result


# In[44]:


if __name__ == '__main__':
    test, test_2 = get_defaultbox(image_size=(40, 40, 3), aspect=[[1,1],[1,2],[1,3]], smin=0.4, smax=1.0, layers=[(5, 5), (3, 3), (1, 1)])
    test = np.array(test)
    #print(test)
    with open('cx_cy_600.csv', 'w') as f:
        for t in test_2:
            txt = '{},{}\n'.format(t[0], t[1])
            f.write(txt)
    #for child in test:
        #print(child)
