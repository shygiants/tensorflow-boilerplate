import tensorflow as tf
from tflibs.model import Model, Network
from tflibs.training import Optimizer, Dispatcher


class DCN(Network):
    def __init__(self, is_chief, features, labels):
        Network.__init__(self, is_chief, features, labels)

        image = features['image']
        batch_size = image.shape.as_list()[0]
        tf.summary.image('Input_Images', image, max_outputs=batch_size, family='Images')

        self._logits = None
        self._loss = None
        self._scores = None
        self._predictions = None

        self._var_defined = False

    def dcn(self, images):
        with tf.variable_scope('ConvNets', values=[images],
                               reuse=None if not self._var_defined and self.is_chief else True):
            layers = tf.layers.conv2d(images, 32, 3, strides=(2, 2), activation=tf.nn.relu)
            outputs = tf.layers.dense(tf.layers.flatten(layers), 10)
            self._var_defined = True
            return outputs

    @property
    def logits(self):
        if self._logits is None:
            self._logits = self.dcn(self.features['image'])

        return self._logits

    @property
    def loss(self):
        if self._loss is None:
            self._loss = tf.losses.softmax_cross_entropy(self.labels, self.logits)

        return self._loss

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


def train(images, labels, beta1, beta2, decay_iters, decay_steps, learning_rate, train_iters, gpu):
    with tf.variable_scope('DeepConvNet', values=[images, labels]):
        global_step = tf.train.get_or_create_global_step()
        optimizer = Optimizer(learning_rate, '', beta1, beta2, train_iters, decay_iters, decay_steps)

        if gpu:
            gpus = map(int, gpu.split(','))
            dispatcher = Dispatcher(DCN, {}, gpus, {'image': images}, labels)

            train_op = dispatcher.minimize(optimizer, lambda dcn: dcn.loss, global_step=global_step)
            loss = dispatcher.chief.loss
        else:
            dcn = DCN(True, {'image': images}, labels)
            loss = dcn.loss
            train_op = optimizer.train_op(dcn.loss, global_step=global_step)

        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)


def evaluate(images, labels):
    with tf.variable_scope('DeepConvNet', values=[images, labels]):
        dcn = DCN(True, {'image': images}, labels)

        metrics = {
            'accuracy': tf.metrics.accuracy(tf.argmax(labels, axis=1), dcn.predictions)
        }

        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.EVAL,
                                          loss=dcn.loss,
                                          eval_metric_ops=metrics)


def predict(images):
    with tf.variable_scope('DeepConvNet', values=[images]):
        dcn = DCN(True, {'image': images}, None)

        predictions = {
            'predictions': dcn.predictions,
        }

        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.estimator.export.PredictOutput(dcn.predictions),
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

        ##########
        # Device #
        ##########
        argparser.add_argument('--gpu',
                               type=str,
                               help='GPU ids for training.',
                               default=None)


export = DeepConvNet
