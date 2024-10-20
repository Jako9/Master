import torch.nn as nn
from .base_network import Large_Network

class Rigid(Large_Network):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env)