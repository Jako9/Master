import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)

from abc import ABC, abstractmethod

def make_env(env_id, seed, idx, dataset, capture_video, run_name, video_path):
    def thunk():
        if "custom" in env_id: # Custom environments
            env = gym.make(f"{env_id.split('/')[-1]}-v0", dataset=dataset)

            if capture_video and idx  ==0:
                env = gym.wrappers.RecordVideo(env, f"{video_path}/{run_name}")

            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = FrameStackEmulator(env, 4)
        else: # Atari environments
            env = gym.make(env_id, render_mode="rgb_array")

            if capture_video and idx  ==0:
                env = gym.wrappers.RecordVideo(env, f"{video_path}/{run_name}")

            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)

            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
            env.action_space.seed(seed)

        return env
    
    return thunk

"""
Abstract class that all environments should inherit from.
This class provides an interface for environments to change internal dynamics.

To implement a new concept drift override the inject_drift method.

If the dýnamic change includes permuting whole classes, self.label_lookup should be used.
"""
class Concept_Drift_Env(gym.Env, ABC):
    def __init__(self, dataset, max_episode_steps = 200) -> None:
        self.data = dataset.data
        self.labels = dataset.targets
        self.shape = dataset.shape
        self.label_lookup = range(np.max(self.labels) + 1)

        self.steps = 0
        self.i = 0
        self.max_episode_steps = max_episode_steps
        self.accumulated_reward = 0

        self.observation_space = spaces.Box(low=0, high=255, shape=(28,28), dtype='uint8')
        self.action_space = spaces.Discrete(10)

        
        self.metadata = {
            'render_modes': ['rgb_array'],
            'render_fps': 10
        }
        self.render_mode = 'rgb_array'
    
    @abstractmethod
    def inject_drift(self):
        raise NotImplementedError
    
    def get_labels(self):
        return self.label_lookup
    
    def step(self, action: int):
        y = self.labels[self.i]

        reward = 0
        done = False

        if action == self.label_lookup[y]:
            reward = 1

        self.steps += 1

        if self.steps >= self.max_episode_steps:
            done = True

        self.i = (self.i + 1) % (len(self.data) - 1)
        
        self.accumulated_reward += reward
        return self.data[self.i], reward, done, done, {}
    
    def reset(self, seed: int = None, options: dict = None):
        np.random.seed(seed)
        self.i = np.random.randint(len(self.data))
        self.steps = 0
        self.accumulated_reward = 0

        return self.data[self.i], {}
    
    def get_action_meanings(self):
            return [str(i) for i in range(len(self.label_lookup))]
    
    def render(self):
        from PIL import Image, ImageDraw, ImageFont

        image = Image.new('L', self.shape, color=0)
        draw = ImageDraw.Draw(image)

        font = ImageFont.load_default()
        reward_as_text = str(self.accumulated_reward)
        bbox = draw.textbbox((0, 0), reward_as_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_position = (14 - text_width // 2, 14 - text_height // 2)
        draw.text(text_position, reward_as_text, fill=255, font=font)

        reward_as_image = np.array(image)
        image = self.data[self.i]

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