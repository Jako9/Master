from ..base_network import ResNet18

class Resnet18_Rigid(ResNet18):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)