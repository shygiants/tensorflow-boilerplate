""" Trainer """

import tensorflow as tf
from tensorflow.python import debug as tfdbg
from tflibs.runner import Runner, DatasetInitializer, TrainInitializer
from tflibs.training import EvaluationRunHook
from tflibs.datasets import build_input_fn
from tflibs.utils import strip_dict_arg
from tflibs.ops import normalize


def run(job_dir,
        train_iters,
        estimator,
        dataset,
        train_batch_size,
        eval_batch_size,
        eval_steps,
        debug,
        debug_address):
    ############
    # Datasets #
    ############
    dataset_train = dataset.read(split='train')
    dataset_test = dataset.read(split='test')

    @strip_dict_arg
    def map_fn(image, label, _id):
        return {
                   'image': normalize(image),
                   '_id': _id,
               }, tf.to_float(label)

    #######
    # Run #
    #######
    try:
        global_step = estimator.get_variable_value('global_step')
    except ValueError:
        global_step = 1

    tf.logging.info('Start training for %d.', global_step)

    hooks = [EvaluationRunHook(estimator,
                               build_input_fn(dataset_test,
                                              eval_batch_size,
                                              map_fn=map_fn,
                                              shuffle_and_repeat=False),
                               eval_steps,
                               summary=False)]

    if debug:
        hooks.append(tfdbg.TensorBoardDebugHook(debug_address))

    # Run training for `train_iters` times
    estimator.train(build_input_fn(dataset_train,
                                   train_batch_size,
                                   map_fn=map_fn,
                                   global_step=global_step,
                                   shuffle_and_repeat=True),
                    max_steps=train_iters,
                    # Run evaluation every `eval_steps` iterations
                    hooks=hooks)


if __name__ == '__main__':
    runner = Runner(initializers=[
        DatasetInitializer(),
        TrainInitializer(),
    ])
    parser = runner.argparser

    ##############
    # Run Config #
    ##############
    parser.add_argument('--eval-steps',
                        type=int,
                        default=5000,
                        help='The number of steps to evaluate the model.')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Whether to debug.')
    parser.add_argument('--debug-address',
                        default='grpc://localhost:6064',
                        help='The address of debugger server.')

    runner.run(run)
