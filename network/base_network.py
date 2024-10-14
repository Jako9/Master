from abc import ABC, abstractmethod
import torch.nn as nn

class Injectable(ABC, nn.Module):
    @abstractmethod
    def inject_plasticity(self):
        raise NotImplementedError

    @abstractmethod
    def check_plasticity_status(self):
        raise NotImplementedError