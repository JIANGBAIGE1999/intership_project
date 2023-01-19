import tensorflow as tf
import numpy as np


INPUT_SIZE = (40, 40, 3)


def conv_bn_relu(x, ch, kernel, stride=1, padding='same', batchnorm=True, relu=True):
    h = tf.keras.layers.Conv2D(ch, kernel, 
                               strides=(stride, stride), 
                               padding=padding)(x)
    if batchnorm:
        h = tf.keras.layers.BatchNormalization()(h)
    if relu:
        h = tf.keras.layers.Activation(activation='relu')(h)
    return h


def down_sample(x, in_ch, out_ch, bn):
    h = conv_bn_relu(x, in_ch, 3, 1, batchnorm=bn)
    h = conv_bn_relu(h, out_ch, 2, 2, 'valid', batchnorm=bn)
    h = conv_bn_relu(h, out_ch, 3, 1, batchnorm=bn)
    return h


def simple_net(x, num, io_ch=32, ratio=2.0, bn=True):
    ch_list = [io_ch]
    for i in range(num):
        t_ch = int(ch_list[i] * ratio)
        ch_list.append(t_ch)
    h = x
    for i in range(num):
        h = down_sample(h, ch_list[i], ch_list[i+1], bn)
    h1, h2, h3 = extra_layers(h)
    return [h1, h2, h3]


def extra_layers(x):
    # 5x5
    h = conv_bn_relu(x, 128, 1, padding='valid', batchnorm=False)
    h = conv_bn_relu(h, 128, 3, 2, batchnorm=False)
    h1 = h
    # 3x3
    h = conv_bn_relu(h, 64, 1, padding='valid', batchnorm=False)
    h = conv_bn_relu(h, 128, 3, padding='valid', batchnorm=False)
    h2 = h
    #1x1
    h = conv_bn_relu(h, 64, 1, padding='valid', batchnorm=False)
    h = conv_bn_relu(h, 128, 3, padding='valid', batchnorm=False)
    h3 = h
    return h1, h2, h3


def output_block(layers, input_size=INPUT_SIZE, num_class=3, aspect=[[1,1],[2,1],[3,1], [1,2], [1,3]], variances=[0.1, 0.1, 0.2, 0.2]):
    """
    出力層
    """
  
    # 特徴マップごとに位置とクラスのスコアマップを生成する
    h_loc = []
    h_conf = []
        
    for i in range(len(layers)):
        h = layers[i]
        loc = conv_bn_relu(h, 4 * (1 + len(aspect)), 3, batchnorm=False, relu=False)
        conf = conv_bn_relu(h, num_class * (1 + len(aspect)), 3, batchnorm=False, relu=False)
        h_loc.append(tf.keras.layers.Flatten()(loc))
        h_conf.append(tf.keras.layers.Flatten()(conf))
    # 各スコアを連結
    all_h_loc = tf.keras.layers.concatenate(h_loc, axis=1)
    all_h_conf = tf.keras.layers.concatenate(h_conf, axis=1)

    # 各スコアを整形
    num_box = all_h_loc.shape[-1] // 4
    all_h_loc = tf.keras.layers.Reshape((num_box, 4))(all_h_loc)
    all_h_conf = tf.keras.layers.Reshape((num_box, num_class))(all_h_conf)
    
    return tf.keras.layers.concatenate([all_h_loc, all_h_conf], axis=2)


def create_model(bn=True):
    """
    学習用モデルを作成し返す

    :param bn: バッチノーマライゼーションの有無
    :return: 学習用モデル
    """

    # モデル構築
    ch = 32  # 畳み込み層の基底チャネル数
    ratio = 2.0  # チャネル増加レート（1回のdown_sampleでratio倍チャネル数が増加）
    block = 2  # U-Netのdown_sample（up_sample）数

    # モデル構築
    inputs = tf.keras.Input(shape=INPUT_SIZE)
    h = conv_bn_relu(inputs, ch, 1, 1, batchnorm=bn)
    h = simple_net(h, block, ch, ratio, bn)
    outputs = output_block(h)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    model = create_model()
    model.summary()