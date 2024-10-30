import torch.nn as nn
from ..base_network import Large_SNN

class Rigid_SNN(Large_SNN):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)