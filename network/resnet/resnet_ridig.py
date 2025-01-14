from ..base_network import ResNet20

class Resnet_Rigid(ResNet20):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)