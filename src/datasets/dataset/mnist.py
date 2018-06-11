"""
    MNIST
"""

import os

from tflibs.datasets import BaseDataset, ImageSpec, LabelSpec


class MNIST(BaseDataset):
    def __init__(self, dataset_dir, split=None):
        self._split = split
        BaseDataset.__init__(self, os.path.join(dataset_dir, 'mnist'))

    @property
    def tfrecord_filename(self):
        return 'mnist.tfrecord' if self._split is None else 'mnist_{}.tfrecord'.format(self._split)

    def _init_feature_specs(self):
        return {
            'image': ImageSpec([28, 28, 1]),
            'label': LabelSpec(10),
        }


export = MNIST
