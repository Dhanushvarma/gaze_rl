import gymnasium as gym

from gaze_rl.utils.logger import log
from gaze_rl.envs.env_wrappers import ImageObservationWrapper, FrameStackWrapper


def make_envs(cfg, training: bool = True):
    if cfg.env.name == "mujoco":

        # Create environment with wrappers as needed
        env = gym.make(cfg.env.env_id)

        # TODO: add wrapper as needed here

        return env
    else:
        raise NotImplementedError
