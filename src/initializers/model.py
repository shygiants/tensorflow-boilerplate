"""
    Model Initializer
"""
import importlib


def select_model(name, *args, **kwargs):
    module = importlib.import_module('{}.{}'.format('models', name))
    print(module)
    return module.export(*args, **kwargs)
