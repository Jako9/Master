from ..base_network import Large_DNN
import torch

class Reset_Network(Large_DNN):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.cache_folder = kwargs["cache_folder"]

        torch.save(self.state_dict(), f"{self.cache_folder}/initial_params.pth")

def every_drift(self, num_drift):
    loaded_params_dict = torch.load(f"{self.cache_folder}/initial_params.pth")

    if "_orig_mod." not in str(self.state_dict().keys()) and "_orig_mod." in str(loaded_params_dict.keys()):
        adjusted_params_dict = {k.replace("_orig_mod.", ""): v for k, v in loaded_params_dict.items()}
        loaded_params_dict = adjusted_params_dict

    self.load_state_dict(loaded_params_dict)