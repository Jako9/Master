from environments.datasets.base import base_dataset
import numpy as np
from torchvision import datasets

class MnistDataset(base_dataset):
    def __init__(self):
        x_train = datasets.MNIST(root=".", train=True, download=True).data.numpy()
        y_train = datasets.MNIST(root=".", train=True, download=True).targets.numpy()

        self.data = x_train.astype(np.uint8)
        self.targets = y_train

        self.shape = (28,28)