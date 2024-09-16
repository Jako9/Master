import gymnasium as gym
from gymnasium import spaces
import numpy as np
from torchvision import datasets

class MnistEnv(gym.Env):
    def __init__(self, render_mode='rgb_array', max_episode_steps=200):
        x_train = datasets.MNIST(root=".", train=True, download=True).data.numpy()
        y_train = datasets.MNIST(root=".", train=True, download=True).targets.numpy()
        self.x_train = x_train.astype(np.uint8)
        self.y_train = y_train
        self.actions = np.zeros_like(y_train)

        self.steps = 0
        self.i = 0
        self.max_episode_steps = max_episode_steps
        self.label_lookup = [0,1,2,3,4,5,6,7,8,9]

        self.observation_space = spaces.Box(low=0, high=255, shape=(28,28), dtype='uint8')
        self.action_space = spaces.Discrete(10)

        self.metadata = {
            'render_modes': ['rgb_array'],
            'render_fps': 10
        }
        self.render_mode = render_mode

    def step(self, action):
        y = self.y_train[self.i]
        self.actions[self.i] = action

        reward = 0
        done = False

        if action == self.label_lookup[y]:
            reward = 1

        self.steps += 1

        if self.steps >= self.max_episode_steps:
            done = True

        self.i = (self.i + 1) % (len(self.x_train) - 1)
        

        return self.x_train[self.i], reward, done, done, {}

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        self.i = np.random.randint(len(self.x_train))
        self.steps = 0

        return self.x_train[self.i], {}
    
    def get_action_meanings(self):
            return ["NOOP" for _ in range(10)]

    def update_labels(self, labels):
        self.label_lookup = labels

    def get_labels(self):
        return self.label_lookup
    
    def render(self):
        import pygame
        canvas = pygame.Surface((28 + 10,28))
        canvas.fill((0, 0, 0))

        for i in range(28):
            for j in range(28):
                if self.x_train[self.i][i][j] != 0:
                    pygame.draw.rect(canvas, (255, 255, 255), (j, i, 1, 1))
        
        prediction = self.actions[self.i]
        for i in range(28):
            for j in range(10):
                if j == prediction:
                    pygame.draw.rect(canvas, (255, 255, 255), (j + 28, i, 1, 1))
                else:
                    pygame.draw.rect(canvas, (125 * (j / 10.0), 125, 125 - (125 * (j / 10.0))), (j + 28, i, 1, 1))
        
        result = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
        return result


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