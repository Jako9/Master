from environments.base import Concept_Drift_Env
import numpy as np
import gymnasium as gym

class No_Drift(Concept_Drift_Env):
    def __init__(self, dataset, max_episode_steps = 200) -> None:
        super().__init__(dataset, max_episode_steps)
    
    def inject_drift(self):
        gym.logger.warn("No drift is being injected")

class Permute_Labels(Concept_Drift_Env):
    def __init__(self, dataset, max_episode_steps = 200) -> None:
        super().__init__(dataset, max_episode_steps)
    
    def inject_drift(self):
        self.label_lookup = np.random.permutation(self.label_lookup)

class Shuffle_Labels(Concept_Drift_Env):
    def __init__(self, dataset, max_episode_steps = 200) -> None:
        super().__init__(dataset, max_episode_steps)
    
    def inject_drift(self):
        np.random.shuffle(self.labels)

class Rotate_Image(Concept_Drift_Env):
    def __init__(self, dataset, max_episode_steps = 200) -> None:
        super().__init__(dataset, max_episode_steps)
    
    def inject_drift(self):
        self.data = np.rot90(self.data, k=1, axes=(1,2))

class Shuffle_Pixels(Concept_Drift_Env):
    def __init__(self, dataset, max_episode_steps = 200) -> None:
        super().__init__(dataset, max_episode_steps)
    
    def inject_drift(self):
        np.random.shuffle(self.data)    


class Add_Classes(Concept_Drift_Env):
    def __init__(self, dataset, max_episode_steps = 200) -> None:
        super().__init__(dataset, max_episode_steps)
        self.all_data = self.data
        self.all_labels = self.labels
        self.num_classes = 0
        self.max_classes = max(self.labels) + 1
    
    def inject_drift(self):
        self.num_classes += 5

        if self.num_classes > self.max_classes:
            gym.logger.warn("All classes have already been added.. No more classes to add")
            return
        
        self.data = self.all_data[self.all_labels < self.num_classes]
        self.labels = self.all_labels[self.all_labels < self.num_classes]
