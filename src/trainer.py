""" Trainer """

import tensorflow as tf
from tflibs.runner import Runner, DatasetInitializer, TrainInitializer
from tflibs.training import EvaluationRunHook
from tflibs.datasets import build_input_fn
from tflibs.utils import strip_dict_arg
from tflibs.ops import normalize


def run(job_dir,
        train_iters,
        estimator,
        dataset,
        batch_size,
        eval_batch_size,
        eval_steps):
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
    # Run training for `train_iters` times
    estimator.train(build_input_fn(dataset_train,
                                   batch_size,
                                   map_fn=map_fn,
                                   global_step=global_step,
                                   shuffle_and_repeat=True),
                    max_steps=train_iters,
                    # Run evaluation every `eval_steps` iterations
                    hooks=[EvaluationRunHook(estimator,
                                             build_input_fn(dataset_test,
                                                            eval_batch_size,
                                                            map_fn=map_fn,
                                                            shuffle_and_repeat=False),
                                             eval_steps,
                                             summary=False)])


if __name__ == '__main__':
    runner = Runner(initializers=[
        DatasetInitializer(),
        TrainInitializer(),
    ])
    parser = runner.argparser

    ##################
    # Input Pipeline #
    ##################
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='Batch size for training')
    parser.add_argument('--eval-batch-size',
                        type=int,
                        default=16,
                        help='Batch size for evaluation')

    ##############
    # Run Config #
    ##############
    parser.add_argument('--eval-steps',
                        type=int,
                        default=5000,
                        help='The number of steps to evaluate the model.')

    runner.run(run)
