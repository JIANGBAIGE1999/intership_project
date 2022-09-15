import math
import numpy as np

def get_iou(bbox1, bbox2):
    x1min, y1min, x1max, y1max = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2min, y2min, x2max, y2max = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    bbox1_area = (x1max-x1min+1)*(y1max-y1min+1) #bbox1の面積
    bbox2_area = (x2max-x2min+1)*(y2max-y2min+1) #bbox2の面積
    x12min=max(x1min, x2min) #intersectionのxmin
    y12min=max(y1min, y2min) #intersectionのymin
    x12max=min(x1max, x2max) #intersectionのxmax
    y12max=min(y1max, y2max) #intersectionのymax
    width=max(0, x12max-x12min+1) #intersectionの幅
    height=max(0, y12max-y12min+1) #intersectionの高さ
    intersect=width*height #intersectinの面積
    iou=intersect/(bbox1_area+bbox2_area-intersect) #iouを取得
    return iou

def non_maximum_suppression(bbox, thresh, score=None, limit=None):
    bbox2=bbox.tolist() #引数のnparrayをlistに変換
    index=list(range(len(bbox2))) #ソート前のインデックス
    score2=score.tolist()
    sorted_bbox=[] #ソート後の配列
    sorted_index=[]
    sorted_score=[]
    for i in range(len(bbox2)): #スコアが高い順にソートしている
        max_index=np.argmax(score2)
        sorted_bbox.append(bbox2[max_index])
        sorted_index.append(max_index)
        sorted_score.append(score2[max_index])
        score2[max_index]=-1 #ソート後の配列に格納したbboxはスコアを-1にする
        """ アルゴリズムは https://kikaben.com/non-max-suppression を参考にした """
    output_index=[]
    output_bbox=[]
    output_score=[]
    delete_index=[] #いらないbboxを削除する用の配列
    i=0
    while len(sorted_bbox)!=0: #sorted_bboxが空になったら抜ける
        output_bbox.append(sorted_bbox[0]) #sorted_bboxで最もスコアが高いbboxをoutputに格納
        output_index.append(sorted_index[0])
        output_score.append(sorted_score[0])
        del sorted_bbox[0] #それをsorted_bboxから消去する
        del sorted_index[0]
        del sorted_score[0]
        if len(sorted_bbox)!=0: #上で削除してもsorted_bboxが空になっていないなら
            while len(sorted_bbox)>i: #iを用いてsorted_bboxの要素数繰り返す(forだとイテラブルオブジェクトが1のときできないためwhileにする)
                iou=get_iou(output_bbox[-1], sorted_bbox[i]) #さっき格納したbboxとsorted_bboxのi番目のbboxのiouを取得する
                if iou>=thresh: #もし閾値より高いなら
                    delete_index.append(i) #削除用配列にインデックスを追加
                i+=1
            i=0
            for i in reversed(delete_index): #上で得られた削除したいsorted_bboxのbboxを削除していく
                del sorted_bbox[i]
                del sorted_index[i]
                del sorted_score[i]
        delete_index.clear() #次のループのために中身を全消去
    return np.array(output_index), np.array(output_score)
                    


def decode(default_bbox, mb_loc, mb_conf, nms_thresh=0.45, score_thresh=0.6, variance=[0.1, 0.2]):
    """ nparrayをlistに変換 """
    list_default_bbox=default_bbox.tolist()
    list_mb_loc=mb_loc.tolist()
    list_mb_conf=mb_conf.tolist()
    """ クラススコア閾値を満たすボックスを抽出 """
    score_bbox=[] #クラススコア閾値を満たすbbox
    score_loc=[] #そのbboxに応じたloc
    score_conf=[] #そのbboxに応じたconf
    for i,arr in enumerate(list_default_bbox): #default_boxの全要素に対して
        score_max=max(list_mb_conf[i]) #bboxの最高スコアを取得する
        if score_max >= score_thresh: #もし最高スコアが閾値以上なら
            score_bbox.append(arr) #閾値を満たす配列に格納
            score_loc.append(list_mb_loc[i])
            score_conf.append(list_mb_conf[i])
            
    """ オフセットよりバウンディングボックスの座標を求める """
    bbox=[] #デコード後の座標を格納する配列
    for i,arr in enumerate(score_bbox): #score_bboxの全要素に対して
        cx=(arr[2]+arr[0])/2 #xの中心
        cy=(arr[3]+arr[1])/2 #yの中心
        w=arr[2]-arr[0] #幅
        h=arr[3]-arr[1]#高さ
        """ バウンディングボックスの計算 """
        decode_cx=cx*(1+(variance[0]*score_loc[i][0]))
        decode_cy=cy*(1+(variance[0]*score_loc[i][1]))
        decode_w=w*math.exp(variance[1]*score_loc[i][2])
        decode_h=h*math.exp(variance[1]*score_loc[i][3])
        bbox.append([decode_cx-w/2, decode_cy-h/2, decode_cx+w/2, decode_h+h/2]) #(xmin, ymin, xmax, ymaxに変換して格納)
    """ クラスごとにボックスを分け、NMSを実行 """
    position=[] #バウンディングボックス位置
    probability=[] #バウンディングボックスクラス確率
    label=[] #バウンディングボックスラベル
    class_bbox=[] #nmsを実行するためにクラスごとに分けるための配列
    class_score=[]
    for i in range(len(mb_conf[0])): #クラス数繰り返す
        for j,arr in enumerate(bbox): #bboxの全要素に対して
            if np.argmax(score_conf[j])==i: #そのbboxのクラスがiなら
                class_bbox.append(arr) #クラス用の配列に格納
                class_score.append(max(score_conf[j]))
        arr_class_bbox=np.array(class_bbox)
        arr_class_score=np.array(class_score)
        arr_index, arr_score=non_maximum_suppression(arr_class_bbox,nms_thresh,arr_class_score) #nmsを実行
        index=arr_index.tolist()
        score=arr_score.tolist()
        for j,arr in enumerate(index): #nmsで得られたインデックスの要素数繰り返す
            position.append(class_bbox[arr]) #出力用の配列に格納する
            probability.append(score[j])
            label.append(i)
        class_bbox.clear() #次のクラスのnmsのために中身を全消去
        class_score.clear()
            
    return np.array(position), np.array(label), np.array(probability)
    
        
    
        

if __name__ == '__main__':
    """ ここにテストコードを作成 """
    default_bbox=np.array([[1.4,2.1,3.6,4.3],[1,2,3,4],[1,2,3,4]])
    mb_conf=np.array([[1.2,2.3,3.3,4.3],[1,2,4,3],[1,2,4.1,3]])
    mb_loc=np.array([[1.1,2.5,3.4,4.6],[1,2,3,4],[1,2,3,4]])
    variance=np.array([0.1,0.2])
    position, label, probability=decode(default_bbox, mb_loc, mb_conf, 0.45, 0.6, variance)
    print(position)
    print(label)
    print(probability)
                      