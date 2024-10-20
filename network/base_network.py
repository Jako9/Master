import torch.nn as nn
from abc import ABC, abstractmethod

"""
Abstract class that all networks should inherit from.
This class provides an interface for hooking on different levels on the training loop.

To implement the fitting plasticity method just override the method corresponding to the correct hook-level.
Also implement the _forward method which is the forward pass of the network.

1) every_init: This method is called once at the beginning of the training loop.
2) every_drift: This method is called after every drift.
3) every_step: This method is called after every step in the environment.
"""
class Plastic(ABC, nn.Module):

    def forward(self, x):
        return self._forward(x / 255.0)
    
    @abstractmethod
    def _forward(self, x):
        raise NotImplementedError
    
    def every_init(self):
        pass

    def every_drift(self, num_drift):
        pass

    def every_step(self, step):
        pass

"""
Standart large network with 3 convolutional layers and 2 linear layers.
It provides interfaces for the three common plasticity methods.

1) Use external modifications: Don't alter anything in the network, just use the external modifications.
2) Modify the networks head: Use self.body and self.head to easily work on the output layer alone.
3) Modify the whole network: Use self.layer__name for each layer to work on the whole network.
"""
class Large_Network(Plastic):
    def __init__(self, env, *args, **kwargs):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2d_2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv2d_3 = nn.Conv2d(64, 64, 3, stride=1)
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.linear = nn.Linear(3136, 512)
        self.head = nn.Linear(512, env.single_action_space.n)

        self.body = nn.Sequential(
            self.conv2d_1,
            self.relu,
            self.conv2d_2,
            self.relu,
            self.conv2d_3,
            self.relu,
            self.flatten,
            self.linear,
            self.relu
        )

    def _forward(self, x):
        x = self.body(x)
        return self.head(x)
    

