from abc import ABC

class base_dataset(ABC):
    def __init__(self):
        self.data = None
        self.targets = None
        
        self.class_names = None

        self.shape = None

    def next(self):
        return self.data, self.targets
    