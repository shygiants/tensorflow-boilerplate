"""
    Dataset Initializer
"""
import importlib


def select_dataset_spec(name, *args, **kwargs):
    module = importlib.import_module('{}.{}'.format('datasets.specs', name))
    print(module)
    return module.export(*args, **kwargs)
