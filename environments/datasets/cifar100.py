from environments.datasets.base import base_dataset
import numpy as np
from torchvision import datasets

class Cifar100Dataset(base_dataset):
    def __init__(self):
        x_train = datasets.CIFAR100(root=".", train=True, download=True).data
        y_train = datasets.CIFAR100(root=".", train=True, download=True).targets

        #Remove color channel from data
        x_train = np.mean(x_train, axis=3)
        self.data = x_train.astype(np.uint8)
        self.targets = np.array(y_train)

        self.class_names = datasets.CIFAR100(root=".", train=True, download=True).classes

        self.shape = (32,32)