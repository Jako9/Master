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
