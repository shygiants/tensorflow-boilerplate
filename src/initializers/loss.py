"""
    Loss Initializer
"""
import tensorflow as tf


def select_loss(name, *args, **kwargs):
    if name == 'categorical_cross_entropy':
        return tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, *args, **kwargs)
    else:
        raise ValueError
