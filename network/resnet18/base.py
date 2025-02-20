from network.base_network import Plastic
from torch import nn

class ResNet18(Plastic):
    from torchvision.models import resnet18
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, env.single_action_space.n)
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=1, padding=1, bias=False)

    def _forward(self, x, global_step):
        return self.model(x)
    