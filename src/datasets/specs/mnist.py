"""
    MNIST
"""
from typing import Dict

from tflibs.dataset import DatasetSpec, FeatureSpec, IDSpec, ImageSpec, LabelSpec
from tflibs.utils import CachedProperty


class MNIST(DatasetSpec):
    @CachedProperty
    def feature_specs(self) -> Dict[str, FeatureSpec]:
        return {
            '_id': IDSpec(),
            'image': ImageSpec([28, 28, 1]),
            'label': LabelSpec(10),
        }


export = MNIST
