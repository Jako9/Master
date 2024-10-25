import torch.nn as nn
from ..base_network import Large_DNN

class Rigid(Large_DNN):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env)