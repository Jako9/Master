from ..base_network import ResNet18SNN

class Resnet18SNN_Rigid(ResNet18SNN):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)