from .base import base_dataset
from .mnist import MnistDataset
from .cifar100 import Cifar100Dataset
from .composite import CompositeDataset

__all__ = ['base_dataset', 'MnistDataset', 'Cifar100Dataset', 'CompositeDataset']