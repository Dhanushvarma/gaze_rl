import gymnasium as gym

from gaze_rl.utils.logger import log


def make_envs(cfg, training: bool = True):
    if cfg.env.name == "mujoco":
        raise NotImplementedError
        # TODO(dhanush): Implement Env Creation for MuJoCo tasks

    else:
        raise NotImplementedError
