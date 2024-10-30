import torch.nn as nn
from ..base_network import Large_DNN

class Reset_Head(Large_DNN):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

    def every_drift(self, num_drift):
        self.head.reset_parameters()
