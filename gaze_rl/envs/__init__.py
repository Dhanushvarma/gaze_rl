from gym.envs.registration import register

register(
    id="pointcross-v0",
    entry_point="gaze_rl.envs.pointcross:PointCrossEnv",
    max_episode_steps=150,
)

register(
    id="pointcrosshard-v0",
    entry_point="gaze_rl.envs.pointcrosshard:PointCrossHardEnv",
    max_episode_steps=150,
)
