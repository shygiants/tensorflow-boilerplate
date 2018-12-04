""" Deep Convolutional Networks """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tflibs.model import Network
from tflibs.nn import conv2d, deconv2d, residual_block, Padding, Nonlinear, Norm, DeconvMethod


class DeepConvNet(Network):
    def __init__(self, scope='DeepConvNet', **hparams):
        Network.__init__(self, **hparams)
        self._scope = scope

        self._scores = None
        self._predictions = None

        self._var_defined = False

    @property
    def scope(self):
        return self._scope

    def __call__(self, inputs):
        with tf.variable_scope(self.scope, values=[inputs], reuse=tf.AUTO_REUSE):
            layers = tf.layers.conv2d(inputs, 32, 3, strides=(2, 2), activation=tf.nn.relu)
            outputs = tf.layers.dense(tf.layers.flatten(layers), 10)

            return outputs



    @property
    def scores(self):
        if self._scores is None:
            self._scores = tf.nn.softmax(self.logits)

        return self._scores

    @property
    def predictions(self):
        if self._predictions is None:
            self._predictions = tf.argmax(self.scores, axis=1)

        return self._predictions
