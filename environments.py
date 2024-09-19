import gymnasium as gym
from gymnasium import spaces
import numpy as np
from torchvision import datasets

from enum import Enum
from abc import ABC, abstractmethod

class Difficulty(Enum):
    EASY = 1
    HARD = 2

class Concept_Drift_Env(gym.Env, ABC):
    @abstractmethod
    def permute_labels(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_labels(self):
        raise NotImplementedError

class MnistEnv(Concept_Drift_Env):
    def __init__(self, render_mode: str = 'rgb_array',
                 difficulty: Difficulty = Difficulty.EASY,
                 max_episode_steps: int = 200):
        
        x_train = datasets.MNIST(root=".", train=True, download=True).data.numpy()
        y_train = datasets.MNIST(root=".", train=True, download=True).targets.numpy()
        self.x_train = x_train.astype(np.uint8)
        self.y_train = y_train
        self.difficulty = difficulty

        self.steps = 0
        self.i = 0
        self.max_episode_steps = max_episode_steps
        self.label_lookup = [0,1,2,3,4,5,6,7,8,9]
        self.accumulated_reward = 0

        self.observation_space = spaces.Box(low=0, high=255, shape=(28,28), dtype='uint8')
        self.action_space = spaces.Discrete(10)

        self.metadata = {
            'render_modes': ['rgb_array'],
            'render_fps': 10
        }
        self.render_mode = render_mode

    def step(self, action: int):
        y = self.y_train[self.i]

        reward = 0
        done = False

        if action == self.label_lookup[y]:
            reward = 1

        self.steps += 1

        if self.steps >= self.max_episode_steps:
            done = True

        self.i = (self.i + 1) % (len(self.x_train) - 1)
        
        self.accumulated_reward += reward
        return self.x_train[self.i], reward, done, done, {}

    def reset(self, seed: int = None, options: dict = None):
        np.random.seed(seed)
        self.i = np.random.randint(len(self.x_train))
        self.steps = 0
        self.accumulated_reward = 0

        return self.x_train[self.i], {}
    
    def get_action_meanings(self):
            return [str(i) for i in range(10)]

    def permute_labels(self):
        if self.difficulty == Difficulty.EASY:
            np.random.shuffle(self.label_lookup)
        else:
            np.random.shuffle(self.y_train)

    def get_labels(self):
        return self.label_lookup
    
    def render(self):
        from PIL import Image, ImageDraw, ImageFont

        image = Image.new('L', (28, 28), color=0)
        draw = ImageDraw.Draw(image)

        font = ImageFont.load_default()
        reward_as_text = str(self.accumulated_reward)
        bbox = draw.textbbox((0, 0), reward_as_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_position = (14 - text_width // 2, 14 - text_height // 2)
        draw.text(text_position, reward_as_text, fill=255, font=font)

        reward_as_image = np.array(image)
        image = self.x_train[self.i]

        result = np.hstack([image, reward_as_image])
        return np.repeat(result[:, :, np.newaxis], 3, axis=2)


class FrameStackEmulator(gym.Wrapper):
     
    def __init__(self, env, k):
        """Repeats the last frame k times to simulate frame stacking.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frame = None
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=((k,)  + obs_shape), dtype=np.uint8)

        self.metadata = {
            'render_modes': ['rgb_array']
        }

    def reset(self, seed=None, options=None):
        ob, info = self.env.reset()
        self.frame = ob
        return self._get_ob(), info

    def step(self, action):
        ob, reward, done, truncated, info = self.env.step(action)
        self.frame = ob
        return self._get_ob(), reward, done, truncated, info

    def _get_ob(self):
        frames = [self.frame] * self.k
        return np.array(frames)