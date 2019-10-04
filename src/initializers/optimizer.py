"""
    Optimizer Initializer
"""
import tensorflow as tf


def select_optimizer(name, *args, **kwargs):
    if name == 'adam':
        return tf.keras.optimizers.Adam(*args, **kwargs)
    else:
        raise ValueError
