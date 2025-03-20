from typing import Dict, Tuple

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import DictConfig
from PIL import Image


from gaze_rl.utils.general_utils import to_numpy
from gaze_rl.utils.logger import log
from gaze_rl.utils.vis_utils import RENDER_META
from r3m import load_r3m


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def flatten_except_first_dim(arr):
    return arr.reshape(-1, *arr.shape[1:])


class FilterTaskStateWrapper(gym.Wrapper):
    def __init__(self, env):
        super(FilterTaskStateWrapper, self).__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        # filter the task low dim state to just get the object poses
        obs["task_low_dim_state"] = obs["task_low_dim_state"][7 * 2 : 7 * 4]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs["task_low_dim_state"] = obs["task_low_dim_state"][7 * 2 : 7 * 4]
        return obs, reward, terminated, truncated, info


class FlattenDictObsWrapper(gym.Wrapper):
    def __init__(self, env, obs_keys, n_envs=1):
        super(FlattenDictObsWrapper, self).__init__(env)
        self.obs_keys = obs_keys
        self.n_envs = n_envs

        # Should be (N, *) and use obs_keys
        if self.n_envs > 1:
            low = np.concatenate(
                [
                    flatten_except_first_dim(env.observation_space[key].low)
                    for key in obs_keys
                ],
                axis=-1,
            )
            high = np.concatenate(
                [
                    flatten_except_first_dim(env.observation_space[key].high)
                    for key in obs_keys
                ],
                axis=-1,
            )
            self.observation_space = spaces.Box(
                low=low, high=high, dtype=env.observation_space[obs_keys[0]].dtype
            )
        else:
            self.observation_space = spaces.Box(
                low=np.concatenate(
                    [env.observation_space[key].low.flatten() for key in obs_keys],
                    axis=-1,
                ),
                high=np.concatenate(
                    [env.observation_space[key].high.flatten() for key in obs_keys],
                    axis=-1,
                ),
                dtype=env.observation_space[obs_keys[0]].dtype,
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        # keep the dictionary in the info for future reference
        info["raw_obs"] = obs
        return self._flatten_obs(obs), info

    def _flatten_obs(self, obs):
        if self.n_envs > 1:
            return np.concatenate(
                [flatten_except_first_dim(obs[key]) for key in self.obs_keys],
                axis=-1,
            )
        return np.concatenate([obs[key].flatten() for key in self.obs_keys])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["raw_obs"] = obs
        return self._flatten_obs(obs), reward, terminated, truncated, info


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, n_envs=1, n_stack=4):
        super(FrameStackWrapper, self).__init__(env)
        self.n_envs = n_envs
        self.n_stack = n_stack
        obs_space = env.observation_space
        self.frames = np.zeros((n_stack,) + obs_space.shape, dtype=np.float32)

        if n_envs > 1:
            axis = 1
        else:
            axis = 0
        self.observation_space = spaces.Box(
            low=flatten_except_first_dim(
                np.repeat(obs_space.low[np.newaxis, ...], n_stack, axis=axis)
            ),
            high=flatten_except_first_dim(
                np.repeat(obs_space.high[np.newaxis, ...], n_stack, axis=axis)
            ),
            dtype=obs_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self.frames.fill(0)
        self.frames[-1] = obs  # Only set the last frame with the initial observation
        return flatten_except_first_dim(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = obs

        return (
            flatten_except_first_dim(self.frames),
            reward,
            terminated,
            truncated,
            info,
        )


class ObsWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset()
        return self._process_obs(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_obs(obs, info)
        return obs, reward, terminated, truncated, info


class ImageObsWrapper(ObsWrapper):
    def __init__(self, env, img_shape: Tuple = (128, 128, 3)):
        super(ImageObsWrapper, self).__init__(env)
        self.image_keys = ["front_rgb"]

        # channel first and normalized
        self.observation_space = spaces.Box(
            low=0, high=1, shape=img_shape, dtype=np.float32
        )

    def _process_obs(self, raw_obs, info):
        obs = info["raw_obs"]["front_rgb"]
        return obs


class FeatureEmbeddingWrapper(ObsWrapper):
    def __init__(self, env, cfg: DictConfig):
        super(FeatureEmbeddingWrapper, self).__init__(env)
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if cfg.model.feature_extractor == "r3m":
            self.feature_extractor = load_r3m("resnet50")
            self.feature_extractor.eval()
            self.feature_extractor.to(self.device)
        else:
            raise NotImplementedError(
                f"Feature extractor {cfg.model.feature_extractor} not implemented"
            )

        # get the feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros((1, 3, 256, 256)).to(self.device)
            self.feature_dim = self.feature_extractor(dummy_input).shape[-1]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.feature_dim,), dtype=np.float32
        )

    def _process_obs(self, obs, info):
        # resize image
        # assuming the image comes in (C, H, W)
        obs = Image.fromarray(obs)

        # R3M expects 256x256 images
        obs = obs.resize((256, 256))
        obs = np.array(obs)

        with torch.no_grad():
            if len(obs.shape) == 3:
                obs = torch.from_numpy(obs[np.newaxis, ...])
                # convert to (N, C, H, W)
                obs = obs.permute(0, 3, 1, 2).float().to(self.device)

            if self.cfg.model.feature_extractor == "r3m":
                # R3M also expects input to be between [0-255], don't normalize the input
                obs = self.feature_extractor(obs)

        return to_numpy(obs.flatten(0, 1))


class NormalizeObsWrapper(ObsWrapper):
    def __init__(self, env):
        super(NormalizeObsWrapper, self).__init__(env)

        img_shape = env.observation_space.shape

        # channel first
        img_shape = (img_shape[2], img_shape[0], img_shape[1])
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=img_shape,
            dtype=np.float32,
        )

    def _process_obs(self, obs, info):
        obs = obs / 255.0

        # make channel first
        obs = obs.transpose(2, 0, 1)
        return obs
