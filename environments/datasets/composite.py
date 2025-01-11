from environments.datasets.base import base_dataset
from environments.datasets.cifar100 import Cifar100Dataset
from torchvision import datasets
import numpy as np

class CompositeDataset(base_dataset):

    class Cifar10(base_dataset):
        def __init__(self):
            x_train = datasets.CIFAR100(root=".", train=True, download=True).data
            y_train = datasets.CIFAR100(root=".", train=True, download=True).targets

            #Remove color channel from data
            x_train = np.mean(x_train, axis=3).astype(np.uint8)

            selected_classes = np.random.choice(np.max(y_train), 10, replace=False)

            self.data = x_train[np.isin(y_train, selected_classes)]
            self.targets = np.array(y_train)[np.isin(y_train, selected_classes)]

            self.targets = np.array([np.where(selected_classes == i)[0][0] for i in self.targets])


            self.class_names = datasets.CIFAR100(root=".", train=True, download=True).classes
            self.class_names = [self.class_names[i] for i in selected_classes]

            self.shape = (32,32)

    def __init__(self, num_objectives=10):
        self.all_datasets = []
        for i in range(num_objectives):
            self.all_datasets.append(self.Cifar10())
        self.current_dataset = 0
        self.data = None
        self.targets = None
        self.shape = self.all_datasets[0].shape
        self.next_objective()

    def next_objective(self):
        self.current_dataset += 1
        self.current_dataset = self.current_dataset % len(self.all_datasets)
        self.data = self.all_datasets[self.current_dataset].data
        self.targets = self.all_datasets[self.current_dataset].targets