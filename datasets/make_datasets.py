import os

ROOT = './data'

datasets = {}
def register_dataset(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator

import torch 

def make_dataset(name, **kwargs):
    """
    Given a dataset name and optional keyword arguments, 
    returns an instance of the corresponding registered dataset class.
    """
    if name not in datasets:
        raise ValueError(f"Unrecognized dataset name '{name}'. Available datasets are: {list(datasets.keys())}")
    if kwargs.get('root') is None:
        kwargs['root'] = os.path.join(ROOT, name)
    dataset = datasets[name](**kwargs)
    return dataset 
