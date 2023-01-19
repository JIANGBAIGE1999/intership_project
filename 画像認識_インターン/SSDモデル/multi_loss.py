# common
import numpy as np

# framework
import tensorflow as tf

def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)

class MultiboxLoss(object):
    def __init__(self, offset, classes, bottom=False, alpha=1, beta=1, k=3):
        self.offset = offset
        self.classes = classes
        self.bottom = bottom
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def hard_negative(self, x, positive, k):
        positive = tf.cast(positive, x.dtype)
        rank = tf.argsort(tf.argsort(x * (positive - 1), axis=1), axis=1)
        negative = tf.expand_dims(tf.reduce_sum(positive, axis=1) * k, -1)

        hard_negative = rank < tf.cast(negative, rank.dtype)
        return hard_negative

    def elementwise_softmax_cross_entropy(self, x, t):
        shape = tf.shape(t)
        # pred -> (BK, 3)
        x = tf.reshape(x, (-1, x.shape[-1]))
        # true -> (BK,) -> OneHot(BK, 3)
        t = tf.cast(tf.reshape(t, (-1,)), tf.int32)
        onehot_labels = tf.one_hot(indices=tf.cast(t, tf.int32), depth=self.classes)

        # maybe, not use tf.losses class
        loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(onehot_labels, x)
        loss = tf.reshape(loss, shape)
        return loss

    def calculate(self, mb_locs, mb_confs, gt_mb_locs, gt_mb_labels):
        # reduce axis=2 (B, K, 1) -> (B, K)
        gt_mb_labels = tf.reduce_sum(gt_mb_labels, axis=2) 
        positive = gt_mb_labels > 0
        n_positive = tf.reduce_sum(tf.cast(positive, tf.int32))

        loc_loss, conf_loss = tf.cond(tf.equal(n_positive, tf.constant(0)), 
                                      lambda: self.return_zero(), 
                                      lambda: self.calc_loss(mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, positive, n_positive))
        return loc_loss, conf_loss
        
    def return_zero(self):
        loss = tf.zeros((), dtype=tf.float32)
        return loss, loss

    def calc_loss(self, mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, positive, n_positive):
        # maybe, not use tf.losses class
        
        loc_loss = huber_loss(gt_mb_locs, mb_locs, clip_delta=1.0)

        # loc_loss = tf.losses.huber_loss(gt_mb_locs, mb_locs, delta=1.0, reduction=tf.losses.Reduction.NONE)
        loc_loss = tf.reduce_sum(loc_loss, axis=-1)
        loc_loss *= tf.cast(positive, loc_loss.dtype)
        loc_loss = tf.reduce_sum(loc_loss) / tf.cast(n_positive, loc_loss.dtype)

        conf_loss = self.elementwise_softmax_cross_entropy(mb_confs, gt_mb_labels)
        
        hard_negative = self.hard_negative(conf_loss, positive, self.k)
        
        conf_loss *= tf.cast(tf.math.logical_or(positive, hard_negative), conf_loss.dtype)
        conf_loss = tf.reduce_sum(conf_loss) / tf.cast(n_positive, conf_loss.dtype)
        return loc_loss, conf_loss

    # True(?, ?, ?)->(B, K, 5) Pred(B, K, 7)
    def multibox_loss(self, y_true, y_pred):
        # loc -> (B, K, 4) conf -> (B, K, 3)
        mb_locs, mb_confs = y_pred[:, :, :self.offset], y_pred[:, :, self.offset:self.offset + self.classes]
        # gt_loc -> (B, K, 4) gt_conf -> (B, K, 1)
        gt_mb_locs, gt_mb_confs = y_true[:, :, :self.offset], y_true[:, :, self.offset:self.offset + 1]
        loc_loss, conf_loss = self.calculate(mb_locs, mb_confs, gt_mb_locs, gt_mb_confs)
        
        loss = loc_loss * self.alpha + conf_loss
        return loss