import gym
from nes_py.wrappers import JoypadSpace
from gym.spaces import Box
from gym.wrappers import FrameStack

import torch
import numpy as np
from torchvision import transforms as T


class SkipFrame(gym.Wrapper):
    """
    Use every n-th frame to represent all frames.
    As consecutive frames don't change very much.
    """
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """
        Repeat action for all n frames and sum reward.
        """
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunc, info


class GrayScaleObservation(gym.ObservationWrapper):
    """
    Transform rgb image to grayscale image
    Lower obs dimension from (3, x, y) to (1, x, y)
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8
        )

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(
            observation.copy(),
            dtype=torch.float
        )
        return observation

    def observation(self, observation):
        """Perform transformation"""
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    """
    Downsample observation into a square image
    """
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8
        )

    def observation(self, observation):
        """Perform transformation"""
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation