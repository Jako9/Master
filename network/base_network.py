import torch.nn as nn


"""
Abstract class that all networks should inherit from.
This class provides an interface for hooking on different levels on the training loop.

To implement the fitting plasticity method just override the method corresponding to the correct hook-level.
"""
class Plastic(nn.Module):

    def forward(self, x):
        return self.network(x / 255.0)
    
    def every_init(self):
        pass

    def every_drift(self, num_drift):
        pass

    def every_step(self, step):
        pass
    

