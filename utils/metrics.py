import tensorflow as tf


def accuracy(y_pred, y):
    check_equal = tf.cast(y_pred == y, tf.float32)
    acc_val = tf.reduce_mean(check_equal)
    return acc_val
