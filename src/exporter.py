""" Exporter """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tflibs.ops import normalize
from tflibs.runner import Runner, ModelInitializer


def run(job_dir,
        model_cls,
        model_args,
        step,
        **kwargs):
    ##########
    # Models #
    ##########
    estimator = tf.estimator.Estimator(
        model_cls.model_fn,
        model_dir=job_dir,
        params={'model_args': model_args})
    tf.logging.info(estimator)

    #######
    # Run #
    #######
    try:
        global_step = estimator.get_variable_value('global_step')
    except ValueError:
        global_step = 1

    tf.logging.info('Export a model for %d.', step or global_step)
    estimator.export_savedmodel(job_dir, model_cls.make_map_fn('predict', **model_args),
                                checkpoint_path=os.path.join(job_dir, 'model.ckpt-{}'.format(step)) if step else None)


if __name__ == '__main__':
    runner = Runner(initializers=[
        ModelInitializer(),
    ])
    parser = runner.argparser

    #########
    # Model #
    #########
    parser.add_argument('--step',
                        type=int,
                        default=None,
                        help='Step to save.')

    runner.run(run)
