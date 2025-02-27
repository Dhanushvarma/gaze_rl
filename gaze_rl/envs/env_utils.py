import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_composite_controller_config


from gaze_rl.utils.logger import log


def make_robosuite_env(cfg):

    _controller = None
    robots = ["Panda"]

    controller_config = load_composite_controller_config(
        controller=_controller,
        robot=robots[0],
    )

    config = {
        "env_name": cfg.env.task_name,
        "robots": robots,
        "controller_configs": controller_config,
    }

    _env = suite.make(
        **config,
        has_renderer=True,
        renderer="mujoco",
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(_env)

    return env


def make_envs(cfg, training: bool = True):
    if cfg.env.name == "robosuite":
        return make_robosuite_env(cfg)
    else:
        raise NotImplementedError
