import torch.nn as nn
from abc import ABC, abstractmethod

"""
Abstract class for all networks.
This class provides an interface for hooking on different levels on the training loop.

To implement the fitting plasticity method just override the method corresponding to the correct hook-level.
Also implement the _forward method which is the forward pass of the network.

1) every_init: This method is called once at the beginning of the training loop.
2) every_drift: This method is called after every drift.
3) every_step: This method is called after every step in the environment.
"""
class Plastic(ABC, nn.Module):

    def __init__(self, total_steps, total_drifts, *args, **kwargs):
        super().__init__()
        self.total_steps = total_steps
        self.total_drifts = total_drifts

    def forward(self, x):
        return self._forward(x / 255.0)
    
    @abstractmethod
    def _forward(self, x):
        raise NotImplementedError

    def init_params(self, num_drifts, num_steps):
        self.total_steps = num_steps
        self.total_drifts = num_drifts
    
    def every_init(self):
        pass

    def every_drift(self, num_drift):
        pass

    def every_step(self, step):
        pass

"""
Standard large deep neural network with 3 convolutional layers and 2 linear layers.
It provides interfaces for the three common plasticity methods.

1) Use external modifications: Don't alter anything in the network, just use the external modifications.
2) Modify the networks head: Use self.body and self.head to easily work on the output layer alone.
3) Modify the whole network: Use self.layer__name for each layer to work on the whole network.
"""
class Large_DNN(Plastic):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
    
import snntorch as snn
import torch
import numpy as np
from snntorch import spikegen

"""
Standard large deep spiking neural network with 3 convolutional layers and 2 linear layers.
It provides interfaces for the three common plasticity methods.

1) Use external modifications: Don't alter anything in the network, just use the external modifications.
2) Modify the networks head: Use self.body and self.head to easily work on the output layer alone.
3) Modify the whole network: Use self.layer__name for each layer to work on the whole network.
"""
class Large_SNN(Plastic):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_steps = 10

        self.action_space = env.single_action_space.n

        self.conv2d_1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2d_2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv2d_3 = nn.Conv2d(64, 64, 3, stride=1)
        
        self.flatten = nn.Flatten()

        self.linear = nn.Linear(3136, 512)
        self.head = nn.Linear(512, self.action_space)

        self.lif1 = snn.Leaky(beta=0.95)
        self.lif2 = snn.Leaky(beta=0.95)
        self.lif3 = snn.Leaky(beta=0.95)
        self.lif_fc = snn.Leaky(beta=0.95)
        #TODO: Implement a real LI head instead of modifying LIF
        self.lif_head = snn.Leaky(beta=0.95, threshold=np.iinfo(np.int32).max)

        self.body = nn.Sequential(
            self.conv2d_1,
            self.lif1,
            self.flatten,
            self.linear,
            self.lif_fc
        )

    def _forward(self, x):

        spike_train = spikegen.rate(x, num_steps=self.num_steps)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_fc = self.lif_fc.init_leaky()
        mem_head = self.lif_head.init_leaky()

        mem_out = torch.zeros(x.size(0), self.action_space).to(x.device)

        for step in range(self.num_steps):
            out = self.conv2d_1(spike_train[step])
            spk1, mem1 = self.lif1(out, mem1)

            out = self.conv2d_2(spk1)
            spk2, mem2 = self.lif2(out, mem2)

            out = self.conv2d_3(spk2)
            spk3, mem3 = self.lif3(out, mem3)

            out = self.flatten(spk3)
            out = self.linear(out)
            spk_fc, mem_fc = self.lif_fc(out, mem_fc)

            out = self.head(spk_fc)

            _, mem_head = self.lif_head(out, mem_head)

            #TODO: Be able to add other pooling methods (mean, last, etc.)

            mem_out = torch.max(mem_head, mem_out)

        return mem_out



    

