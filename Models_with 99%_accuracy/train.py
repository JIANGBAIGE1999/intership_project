import os
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import keras

from dataset import DataSequence
from model import create_model
from param import get_param

def main():

    # 各種パラメータを取得
    param = get_param()

    part_class =param['target_part_class']
    non_dir = os.path.join(part_class,'non')
    def_dir = os.path.join(part_class,'def')
    
    # 画像名を取得
    non_names = os.listdir(non_dir)
    def_names = os.listdir(def_dir)

    # 画像パスを生成
    non_path = [os.path.join(non_dir, n) for n in non_names]
    def_path = [os.path.join(def_dir, n) for n in def_names]

    # データ数を取得
    num_p = len(non_path)
    num_f = len(def_path)

    # 訓練データと検証データを分割
    split_idx_p = int(num_p * param['data_split_rate'])
    split_idx_f = int(num_f * param['data_split_rate'])

    # 訓練用、検証用の画像パスを取得
    train_x = np.array(non_path[:split_idx_p] + def_path[:split_idx_f])
    val_x =   np.array(non_path[split_idx_p:] + def_path[split_idx_f:])

    # ラベル生成　傷無：0, 傷有:1
    label_p = np.zeros(num_p)
    label_f = np.ones(num_f)

    # 訓練用、検証用のラベルデータを取得
    train_y = np.concatenate([label_p[:split_idx_p], label_f[:split_idx_f]], axis=-1)
    val_y =   np.concatenate([label_p[split_idx_p:], label_f[split_idx_f:]], axis=-1)

    # データジェネレータを生成
    train_gen = DataSequence(train_x, train_y, param, param['train_augment_flag'])
    val_gen   = DataSequence(val_x, val_y, param, param['val_augment_flag'])

    # CNNモデル生成
    model = create_model(input_shape=param['input_shape'], 
                         first_ch=param['first_conv_ch'], 
                         num_block=param['num_block'], 
                         n_classes=param['n_classes']) 
    # オプティマイザ設定
    opt = keras.optimizers.SGD(learning_rate=param['learning_rate'], 
                               momentum=param['momentum'])

    # モデルコンパイル
    model.compile(optimizer=opt,
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])

    def scheduler(epoch, lr):
        """
        学習率スケジューラ
        """
        if epoch < param['shift_epoch'][0]:
            return lr
        elif epoch < param['shift_epoch'][1]:
            return lr * param['shift_learning_rate']
        else:
            return lr *  param['shift_learning_rate'] ** 2

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # モデルサマリ表示
    model.summary()

    # 学習スタート
    history = model.fit_generator(train_gen, 
                                  epochs=param['train_epoch'], 
                                  validation_data=val_gen, 
                                  shuffle=True, 
                                  class_weight=param['class_weight'], 
                                  callbacks=[callback])
    
    # モデル保存
    model.save(param['model_path'])

    # ログフォルダ作成
    os.makedirs(param['output_path'], exist_ok=True)

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    acc_fig_path = os.path.join(param['output_path'], 'acc.png')
    plt.savefig(acc_fig_path)

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    loss_fig_path = os.path.join(param['output_path'], 'loss.png')
    plt.savefig(loss_fig_path)



if __name__ == '__main__':
    main()
