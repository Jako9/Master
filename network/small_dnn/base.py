from torch import nn
from network.base_network import Plastic

class Small_DNN(Plastic):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv2d_1 = nn.Conv2d(4, 64, 8, stride=2)
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.linear = nn.Linear(97344, 512)
        self.head = nn.Linear(512, env.single_action_space.n)

        self.body = nn.Sequential(
            self.conv2d_1,
            self.relu,
            self.flatten,
            self.linear,
            self.relu
        )

    def _forward(self, x, global_step):
        x = self.body(x)
        return self.head(x)