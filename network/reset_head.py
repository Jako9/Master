import torch.nn as nn
from .base_network import Large_Network

class Reset_Head(Large_Network):
    def __init__(self, env, *args, **kwargs):
        super().__init__()

    def every_drift(self, num_drift):
        self.head.reset_parameters()
