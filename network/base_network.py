import torch.nn as nn
from abc import ABC, abstractmethod
import inspect
import sys
import os
import importlib.util
from network.generic_modifiers import *

"""
Abstract class for all networks.
This class provides an interface for hooking on different levels on the training loop.

To implement the fitting plasticity method just override the method corresponding to the correct hook-level.
Also implement the _forward method which is the forward pass of the network.

1) every_init: This method is called once at the beginning of the training loop.
2) every_drift: This method is called after every drift.
3) every_step: This method is called after every step in the environment.
"""
class Plastic(ABC, nn.Module):

    def __init__(self, total_steps, total_drifts, track, *args, **kwargs):
        super().__init__()
        self.track = track
        self.total_steps = total_steps
        self.total_drifts = total_drifts

    def forward(self, x, global_step=None):
        return self._forward(x / 255.0, global_step)
    
    @abstractmethod
    def _forward(self, x, global_step):
        raise NotImplementedError

    def init_params(self, num_drifts, num_steps):
        self.total_steps = num_steps
        self.total_drifts = num_drifts
    
    def every_init(self):
        pass

    def every_drift(self, num_drift):
        pass

    def every_step(self, step):
        pass

def build_networks():
    implemented_classes = get_implemented_classes()
    generic_classes = build_generic_modifiers()
    classes = implemented_classes + generic_classes
    class_dict = {cls.__name__: cls for cls in classes}
    return class_dict

def get_implemented_classes():
    root_dir = "network"
    classes = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".py") and file != "base.py" and dirpath != root_dir:
                file_path = os.path.join(dirpath, file)
                module_name = file_path.replace('/', '.').rstrip('.py')
                #get classes from module name
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for obj in module.__dict__.values():
                    if inspect.isclass(obj) and issubclass(obj, Plastic) and obj.__name__ != "Plastic" and Plastic not in obj.__bases__:
                        classes.append(obj)

    return classes

def build_generic_modifiers():
    def find_base_files(root_dir):
        """Find all base.py files in subdirectories."""
        base_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            if 'base.py' in filenames:
                base_files.append(os.path.join(dirpath, 'base.py'))

        return base_files
    
    def get_class_from_base(file_path):
        """Dynamically import and return the class from base.py"""
        module_name = file_path.replace('/', '.').rstrip('.py')  # Convert path to module notation
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Extract class (assuming only one class in base.py)
        for obj in module.__dict__.values():
            # Check if it's a class and not the parent class Plastic
            if isinstance(obj, type) and obj.__name__ != "Plastic" and issubclass(obj, Plastic):
                return obj
        return None
    
    generic_modifiers = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) and not obj.__name__ == "Plastic":
            if "__init__" in obj.__dict__:
                generic_modifiers.append(obj)

    base_files = find_base_files("network")
    base_classes = []
    
    for file in base_files:
        base_class = get_class_from_base(file)
        if base_class is not None:
            base_classes.append(base_class)

    classes = []
    for modifier in generic_modifiers:
        for base_class in base_classes:
            #Build a class of modifier that inherits from the base class
            class_name = f"{base_class.__name__}_{modifier.__name__}"
            new_class = type(class_name, (modifier, base_class), {})
            classes.append(new_class)
    
    return classes


