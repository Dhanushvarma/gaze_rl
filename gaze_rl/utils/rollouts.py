import collections
import time
from typing import Dict, Union

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import tqdm
from omegaconf import DictConfig

import wandb
from gaze_rl.utils.general_utils import to_numpy
from gaze_rl.utils.logger import log
from gaze_rl.utils.vis_utils import annotate_single_video


def run_eval_rollouts(
    cfg: DictConfig, envs, model: nn.Module, wandb_run=None, device: torch.device = None
):
    """
    Run evaluation rollouts for RoboSuite environments
    """
    rollout_videos = []
    error_stats = {"UnknownError": 0}
    eval_metrics = collections.defaultdict(float)
    rollouts = []

    num_videos_render = cfg.num_eval_rollouts_render

    for rollout_indx in tqdm.tqdm(
        range(cfg.num_eval_rollouts), desc="running rollouts"
    ):
        log_video = num_videos_render > 0
        metrics, rollout = _perform_single_rollout(
            cfg, envs, model, log_videos=log_video, device=device
        )

        # This means the rollout failed
        if rollout is None:
            error_stats[metrics.get("error", "UnknownError")] += 1
        else:
            rollouts.append(rollout)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    eval_metrics[k] += v

            if log_video:
                num_videos_render -= 1

                T = rollout["observation"].shape[0]

                annotations = {
                    "S": np.arange(T),
                    "Ret": rollout["reward"].cumsum(axis=0),
                    "R": rollout["reward"],
                    "D": rollout["done"],
                    "A": rollout["action"],
                }

                # RoboSuite specific annotations
                env_ann = {}
                if "gripper_pos" in rollout["info"]:
                    env_ann["Gripper_Pos"] = rollout["info"]["gripper_pos"]
                if "object_pos" in rollout["info"]:
                    env_ann["Object_Pos"] = rollout["info"]["object_pos"]
                    
                annotations.update(env_ann)

                annotated_video = annotate_single_video(
                    video=rollout["image"],
                    annotations=annotations,
                    subgoal_vars_per_dim=metrics.get("subgoal_vars_per_dim", []),
                )
                rollout_videos.append(annotated_video)

    # average metrics
    for k, v in eval_metrics.items():
        eval_metrics[k] /= cfg.num_eval_rollouts

    eval_metrics.update(error_stats)

    if wandb_run is not None:
        if len(rollout_videos) <= 0:
            log("No rollout videos to log to wandb", color="yellow")
        else:
            rollout_videos = np.array(rollout_videos)
            rollout_videos = einops.rearrange(rollout_videos, "n t h w c -> n t c h w")
            wandb_run.log(
                {"rollout_videos/": wandb.Video(rollout_videos, fps=cfg.video_fps)}
            )

    return eval_metrics, rollouts


def compute_subgoal_var(all_sampled_subgoals, keys: str = "gripper_xyz"):
    """Compute variance of subgoals"""
    if keys == "all":
        return np.var(all_sampled_subgoals, axis=0)
    elif keys == "gripper_xyz":
        return np.var(all_sampled_subgoals[:, :3], axis=0)
    elif keys == "robosuite_eef":
        # For robosuite, typically first 3 dimensions are end-effector position
        return np.var(all_sampled_subgoals[:, :3], axis=0)

    raise NotImplementedError(f"Key {keys} not supported for subgoal variance calculation")


def _perform_single_rollout(
    cfg: DictConfig,
    env,
    model: nn.Module,
    log_videos: bool = False,
    device: torch.device = torch.device("cuda"),
) -> Union[Dict[str, np.ndarray], Dict]:
    """
    Perform a single rollout in RoboSuite environment
    
    Return:
        eval_metrics: Dict of evaluation metrics
        rollout: Dict of rollout data
    """
    log(f"Running RoboSuite rollout, log videos: {log_videos}", color="green")

    rollout_start = time.time()

    ep_return = 0
    ep_len = 0
    ep_success = False
    ep_done = False
    subgoal_vars_per_dim = []

    rollout = collections.defaultdict(list)
    hidden = None

    # Reset the environment
    obs, info = env.reset()

    while not ep_done:
        # break after max timesteps
        if ep_len >= cfg.env.max_episode_steps:
            break

        obs_tensor = torch.from_numpy(obs).to(device).float()

        # add batch dimension if needed
        if obs_tensor.shape[0] != 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # Handle hierarchical policies with subgoal sampling
        if cfg.name == "hbc" and ep_len % cfg.rollout.hl_horizon == 0:
            # Sample new subgoal every H steps if using hierarchical policy
            selected_subgoal, all_sampled_subgoals = model.get_subgoal(
                cond=obs_tensor,
                N=cfg.rollout.num_subgoals_sample,
                subgoal_selection="random",
            )
            all_sampled_subgoals = to_numpy(all_sampled_subgoals)
            subgoal_var_per_dim = np.var(all_sampled_subgoals, axis=0)

            # compute a scalar value for subgoal variance
            subgoal_var = compute_subgoal_var(
                all_sampled_subgoals, keys=cfg.rollout.get("subgoal_var_key", "robosuite_eef")
            )
            
            subgoal_vars_per_dim.append(subgoal_var_per_dim)
        else:
            selected_subgoal, all_sampled_subgoals = None, None

        # Get action from policy
        if cfg.name == "bc":
            action_pred = model.get_action(obs_tensor)
        else:
            # RNN-based policy
            action_pred, hidden = model.get_action(
                obs_tensor, subgoal=selected_subgoal, hidden=hidden
            )
            
        action = to_numpy(action_pred)[0]

        # Step the environment
        try:
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Check for success based on task completion
            if 'success' in info:
                ep_success = ep_success or info['success']
                
            # Update observation
            obs = next_obs
        except Exception as e:
            log(f"Environment error: {str(e)}", color="red")
            return {"error": str(e)}, None

        # Update episode stats
        ep_done = terminated or truncated
        ep_return += reward
        ep_len += 1

        # Generate image frames for the video if needed
        if log_videos:
            try:
                # Render the environment
                image = env.render()
                
                # Handle different rendering outputs
                if isinstance(image, np.ndarray):
                    # Direct image from render
                    pass
                elif hasattr(image, 'read'):
                    # File-like object
                    image_data = image.read()
                    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                else:
                    # Fallback to empty frame
                    image = np.zeros((128, 128, 3), dtype=np.uint8)
                    
                # Add visualization of subgoals if available
                if all_sampled_subgoals is not None and hasattr(env, 'visualize_points'):
                    image = env.visualize_points(
                        image, 
                        points=all_sampled_subgoals[:, :3],
                        selected_point=to_numpy(selected_subgoal)[:3] if selected_subgoal is not None else None
                    )
            except Exception as e:
                log(f"Rendering error: {str(e)}", color="yellow")
                image = np.zeros((128, 128, 3), dtype=np.uint8)
        else:
            image = None

        # Store transition data
        rollout["observation"].append(obs)
        rollout["action"].append(action)
        rollout["reward"].append(reward)
        rollout["done"].append(terminated)
        rollout["truncated"].append(truncated)
        rollout["image"].append(image)
        rollout["info"].append(info)

    rollout_time = time.time() - rollout_start

    # Convert rollout lists to numpy arrays
    for k, v in rollout.items():
        if k != "info" and not isinstance(v[0], dict) and v[0] is not None:
            rollout[k] = np.array(v)

    # Process info dictionaries
    infos = rollout["info"]
    processed_info = {}
    
    # Find common keys across all info dicts
    common_keys = set(infos[0].keys())
    for info in infos[1:]:
        common_keys = common_keys.intersection(set(info.keys()))
    
    # Extract these common keys
    for key in common_keys:
        try:
            values = [info[key] for info in infos]
            if all(isinstance(v, (int, float, bool, np.number)) for v in values):
                processed_info[key] = np.array(values)
        except (ValueError, TypeError):
            pass  # Skip if can't convert to array
            
    rollout["info"] = processed_info

    eval_metrics = {
        "ep_return": ep_return,
        "ep_success": float(ep_success),  # Convert boolean to float for averaging
        "ep_len": ep_len,
        "time": rollout_time
    }
    
    if len(subgoal_vars_per_dim) > 0:
        eval_metrics["subgoal_vars_per_dim"] = subgoal_vars_per_dim
        
    return eval_metrics, rollout