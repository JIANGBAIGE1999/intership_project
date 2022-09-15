import tensorflow as tf
import keras

def conv2d(x, f, k, s=1, p='same', bn=True, act=True, name=None):
    """ ２次元の畳み込み関数 """
    if name is None:
        h = keras.layers.Conv2D(f, k, strides=s, padding=p)(x)
    else:
        h = keras.layers.Conv2D(f, k, strides=s, padding=p, name=name)(x)
    if bn:
        h = keras.layers.BatchNormalization()(h)
    if act:
        h = keras.layers.Activation('relu')(h)
    return h

def create_model(input_shape=(512,512, 3), first_ch=32, num_block=5, n_classes=2):
    """ CNNモデル """
    # 畳み込みチャネル数を計算
    channels = [first_ch * (2 ** i) for i in range(num_block)]

    # モデルの入力サイズを設定
    inputs = keras.layers.Input(shape=input_shape)
    h = inputs
    
    # 畳み込みレイヤを生成
    for ch in channels:
        h = conv2d(h, ch, 3, 1) 
        h = keras.layers.MaxPool2D(strides=2, padding='same')(h)

    # 出力層を生成
    h = conv2d(h, n_classes, 3, 1, bn=True, act=False, name='heatmap')
    h = keras.layers.GlobalAveragePooling2D()(h)
    h = keras.layers.Activation('softmax')(h)
    return keras.Model(inputs, h)