import torch.nn as nn
from network.large_dnn.base import Large_DNN

class Layer_Norm(Large_DNN):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.layer_norm_1 = nn.LayerNorm([32, 20, 20])
        self.layer_norm_2 = nn.LayerNorm([64, 9, 9])
        self.layer_norm_3 = nn.LayerNorm([64, 7, 7])
        self.layer_norm_4 = nn.LayerNorm(512)

    def _forward(self, x, global_step):
        x = self.relu(self.layer_norm_1(self.conv2d_1(x)))
        x = self.relu(self.layer_norm_2(self.conv2d_2(x)))
        x = self.relu(self.layer_norm_3(self.conv2d_3(x)))
        x = self.flatten(x)
        x = self.relu(self.layer_norm_4(self.linear(x)))
        return self.head(x)

    
