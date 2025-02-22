import torch.nn as nn
from network.large_dnn.base import Large_DNN

class Large_DNN_Reset_Head(Large_DNN):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

    def every_drift(self, num_drift):
        self.head.reset_parameters()
