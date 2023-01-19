
def get_param():
    """
    パラメータ設定
    """
    param = {}

    # training param
    param['batch_size']          = 4         # バッチサイズ
    param['input_size']          = (512, 512) # 入力画像サイズ
    param['learning_rate']       = 0.01       # 学習率
    param['momentum']            = 0.9        # SGDのモーメンタム
    param['shift_learning_rate'] = 0.1        # 学習率の変化率
    param['shift_epoch']         = [50, 75]   # 学習率の変化エポック数
    param['train_epoch']         = 100        # 学習回数
    param['class_weight']        = {0: 1.0,   # 0: 傷無データの重み
                                    1: 1.0}   # 1: 傷有データの重み

    # validation param

    # model param
    param['model_path']    = './class1.hdf5'  # 保存時のモデル名
    param['input_shape']   = (512, 512, 3)    # CNNの入力サイズ
    param['first_conv_ch'] = 32               # CNNの最初の畳み込みチャネル数
    param['num_block']     = 5                # 畳み込みブロック数
    param['n_classes']     = 2                # クラス数
    
    # dataset
    param['target_part_class']  = 'Class1'    # 学習対象とする部品の種類
    param['data_split_rate']    = 0.8         # 訓練用、検証用の画像分割率　訓練用：検証用 = x : (1 - x)
    param['train_augment_flag'] = False       # 訓練用画像に対するデータ拡張の有無
    param['val_augment_flag']   = False       # 検証用画像に対するデータ拡張の有無
    param['horizonal_flip']     = True        # データ拡張：画像の左右反転
    param['vertical_flip']      = True        # データ拡張：画像の上下反転
    param['rotation']           = True        # データ拡張：画像の回転 (0 ~ 20[deg])

    # preprocess param
    param['normalization'] = True             # 入力画像の正規化の有無
    param['gray_scale']    = False            # グレースケール化の有無

    # training log param
    param['output_path'] = 'log'              # ログ保存用フォルダ名

    return param