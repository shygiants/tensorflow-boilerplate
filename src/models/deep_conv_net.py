import tensorflow as tf
from tflibs.model import Model
from tflibs.training import Optimizer


def train(images, labels, beta1, beta2, decay_iters, decay_steps, learning_rate, train_iters):
    with tf.variable_scope('DeepConvNet', values=[images, labels]):
        global_step = tf.train.get_or_create_global_step()
        optimizer = Optimizer(learning_rate, '', beta1, beta2, train_iters, decay_iters, decay_steps)

        layers = tf.layers.conv2d(images, 32, 3, strides=(2, 2), activation=tf.nn.relu)
        outputs = tf.layers.dense(tf.layers.flatten(layers), 10)

        loss = tf.losses.softmax_cross_entropy(labels, outputs)

        grads = optimizer.compute_grad(loss)
        train_op = optimizer.apply_gradients(grads, global_step=global_step)

        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN, loss=loss,
                                          train_op=train_op)


def evaluate(images, labels):
    with tf.variable_scope('DeepConvNet', values=[images, labels]):
        layers = tf.layers.conv2d(images, 32, 3, strides=(2, 2), activation=tf.nn.relu)
        outputs = tf.layers.dense(tf.layers.flatten(layers), 10)
        outputs = tf.nn.softmax(outputs)
        predictions = tf.argmax(outputs, axis=1)

        metrics = {
            'accuracy': tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions)
        }

        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.EVAL,
                                          loss=tf.zeros([1]),
                                          eval_metric_ops=metrics)


def predict(images):
    with tf.variable_scope('DeepConvNet', values=[images]):
        layers = tf.layers.conv2d(images, 32, 3, strides=(2, 2), activation=tf.nn.relu)
        outputs = tf.layers.dense(tf.layers.flatten(layers), 10)
        outputs = tf.nn.softmax(outputs)
        predictions = tf.argmax(outputs, axis=1)

        predictions = {
            'predictions': predictions,
        }

        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.estimator.export.PredictOutput(predictions),
        }

        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT,
                                          predictions=predictions,
                                          export_outputs=export_outputs)


class DeepConvNet(Model):
    @staticmethod
    def model_fn(features, labels, mode, params):
        images = features['image']
        model_args = params['model_args'] if 'model_args' in params else {}

        if mode == tf.estimator.ModeKeys.TRAIN:
            model_args.update(params['train_args'])
            return train(images, labels, **model_args)
        elif mode == tf.estimator.ModeKeys.EVAL:
            model_args.update(params['eval_args'])
            return evaluate(images, labels)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            return predict(images)
        else:
            raise ValueError

    @classmethod
    def add_model_args(cls, argparser, parse_args):
        pass

    @classmethod
    def add_train_args(cls, argparser, parse_args):
        #############
        # Optimizer #
        #############
        argparser.add_argument('--beta1',
                               type=float,
                               default=0.5,
                               help='Beta1 for Adam optimizer.')
        argparser.add_argument('--beta2',
                               type=float,
                               default=0.999,
                               help='Beta1 for Adam optimizer.')

        ############
        # LR Decay #
        ############
        argparser.add_argument('--decay-iters',
                               type=int,
                               default=100000,
                               help='The number of training iterations to decay learning rate.')
        argparser.add_argument('--decay-steps',
                               type=int,
                               default=1000,
                               help='The number of training steps to decay learning rate.')
        ############
        # Training #
        ############
        argparser.add_argument('--learning-rate',
                               type=float,
                               default=0.0001,
                               help='Learning rate.')

    @classmethod
    def add_eval_args(cls, argparser, parse_args):
        pass


export = DeepConvNet
