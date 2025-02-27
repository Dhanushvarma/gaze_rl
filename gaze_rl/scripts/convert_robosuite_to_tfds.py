"""
Script to convert robosuite dataset from HDF5 to TFDS format.
"""

import argparse
import collections
import os
import json
from pathlib import Path
from typing import List, Dict, Any

import h5py
import numpy as np
import tqdm
import tensorflow as tf

from gaze_rl.scripts.save_data import save_dataset
from gaze_rl.utils.logger import log


def load_robosuite_data(args):
    """
    Load robosuite data from HDF5 file
    """
    data_path = Path(args.data_dir) / args.dataset_file
    
    with h5py.File(data_path, 'r') as f:
        # Extract metadata
        total_samples = f['data'].attrs['total']
        env_args = json.loads(f['data'].attrs['env_args'])
        
        log(f"Loading dataset with {total_samples} total samples", "green")
        log(f"Environment: {env_args.get('env_name')}", "green")
        
        trajectories = []
        
        # Get list of demos (assuming they follow pattern 'demo_X')
        demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
        
        # Filter demos if mask is provided
        if 'mask' in f and args.filter_key in f['mask']:
            demo_keys = list(f['mask'][args.filter_key])
            log(f"Using filter key '{args.filter_key}' with {len(demo_keys)} trajectories")
        
        log(f"Found {len(demo_keys)} trajectories to process")
        
        for demo_key in tqdm.tqdm(demo_keys, desc="Processing trajectories"):
            demo = f['data'][demo_key]
            
            # Number of timesteps in this trajectory
            num_samples = demo.attrs['num_samples']
            
            # Initialize arrays for this trajectory
            observations = []
            actions = []
            rewards = []
            dones = []
            subgoals = [] if args.compute_subgoals else None
            image_dict = collections.defaultdict(list) if args.save_imgs else None
            
            # Get observations, actions, rewards, dones
            obs_dict = {}
            
            # Process observation keys
            for obs_key in demo['obs'].keys():
                if obs_key in args.obs_keys:
                    obs_data = demo['obs'][obs_key][:]
                    obs_dict[obs_key] = obs_data
            
            # Get actions, rewards, dones
            actions_data = demo['actions'][:]
            rewards_data = demo['rewards'][:]
            dones_data = demo['dones'][:]
            
            # Process according to observation processing strategy
            for i in range(num_samples):
                # Combine observations based on args.obs_keys
                observation_parts = []
                
                for key in args.obs_keys:
                    if key in obs_dict:
                        if key in args.image_keys and args.save_imgs:
                            # Store image separately
                            if image_dict is not None and i < len(obs_dict[key]):
                                image_dict[key].append(obs_dict[key][i])
                        else:
                            # Flatten and add to observation vector
                            if i < len(obs_dict[key]):
                                obs_value = obs_dict[key][i].flatten()
                                observation_parts.append(obs_value)
                
                # Concatenate all observation parts
                if observation_parts:
                    observations.append(np.concatenate(observation_parts))
                else:
                    observations.append(np.array([0.0]))  # Empty observation as a placeholder
                
                # Add action, reward, and done if available
                if i < len(actions_data):
                    actions.append(actions_data[i])
                else:
                    actions.append(np.zeros_like(actions_data[0]))
                    
                if i < len(rewards_data):
                    rewards.append(rewards_data[i])
                else:
                    rewards.append(0.0)
                    
                if i < len(dones_data):
                    dones.append(dones_data[i])
                else:
                    dones.append(False)
                
                # For computing subgoals if needed
                if args.compute_subgoals:
                    subgoal_parts = []
                    for key in args.subgoal_keys:
                        if key in obs_dict and i < len(obs_dict[key]):
                            subgoal_parts.append(obs_dict[key][i].flatten())
                    
                    if subgoal_parts:
                        subgoals.append(np.concatenate(subgoal_parts))
                    else:
                        subgoals.append(np.array([0.0]))  # Empty subgoal as a placeholder
            
            # Convert lists to numpy arrays
            observations_array = np.array(observations)
            actions_array = np.array(actions)
            rewards_array = np.array(rewards)
            dones_array = np.array(dones)
            
            # Build the trajectory dictionary
            trajectory = {
                "observations": observations_array,
                "actions": actions_array,
                "rewards": rewards_array,
                "discount": np.ones_like(rewards_array),
                "is_first": np.zeros(num_samples, dtype=bool),
                "is_last": np.zeros(num_samples, dtype=bool),
                "is_terminal": dones_array.astype(bool),
            }
            
            # Mark first state
            if num_samples > 0:
                trajectory["is_first"][0] = True
            
            # Mark last state if not already terminal
            if num_samples > 0 and not any(dones_array):
                trajectory["is_last"][-1] = True
            else:
                # Mark all terminal states as last
                for i, is_terminal in enumerate(dones_array):
                    if is_terminal:
                        trajectory["is_last"][i] = True
            
            # Add subgoals if computed
            if args.compute_subgoals and subgoals:
                trajectory["subgoals"] = np.array(subgoals)
            
            # Add images if saved
            if args.save_imgs and image_dict:
                # Convert each image array to numpy array with consistent shape
                trajectory["images"] = {}
                for k, v in image_dict.items():
                    if v:  # Only add if there are images
                        trajectory["images"][k] = np.array(v)
            
            trajectories.append(trajectory)
    
    return trajectories


def main(args):
    data_dir = Path(args.data_dir)
    returns = []
    lengths = []
    
    # Load robosuite data
    trajectories = load_robosuite_data(args)
    num_trajs = len(trajectories)
    log(f"Loaded Robosuite data, found {num_trajs} trajectories. Saving to RLDS now", "green")
    
    # Calculate statistics
    for traj in trajectories:
        returns.append(np.sum(traj["rewards"]))
        lengths.append(len(traj["rewards"]))
    
    log(f"Number of trajectories: {len(trajectories)}")
    log(f"Average return: {np.mean(returns)}")
    log(f"Average length: {np.mean(lengths)}")
    
    # Generate task name from dataset if not provided
    task = args.task
    if not task and args.dataset_file:
        task = Path(args.dataset_file).stem
    
    save_file = Path(data_dir) / "tensorflow_datasets" / args.env_name / task
    save_dataset(
        trajectories,
        save_file,
        env_name=args.env_name,
        save_imgs=args.save_imgs,
        image_keys=args.image_keys,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Robosuite data to TFDS")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/path/to/robosuite/data",
        help="Path to the robosuite data directory",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="dataset.hdf5",
        help="Name of the HDF5 dataset file",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Name of the task (subdirectory for output). If empty, uses dataset filename.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="robosuite",
        help="Name of the environment",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default="",
        help="Filter key to use from mask group (e.g., 'train', 'valid')",
    )
    parser.add_argument(
        "--save_imgs",
        action="store_true",
        help="Whether to save images",
    )
    parser.add_argument(
        "--compute_subgoals",
        action="store_true",
        help="Whether to compute subgoals from observations",
    )
    parser.add_argument(
        "--obs_keys",
        nargs="+",
        default=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        help="Observation keys to include",
    )
    parser.add_argument(
        "--subgoal_keys",
        nargs="+",
        default=["robot0_eef_pos", "robot0_gripper_qpos"],
        help="Observation keys to use for subgoals",
    )
    parser.add_argument(
        "--image_keys",
        nargs="+",
        default=["agentview_image"],
        help="Image keys to include when save_imgs is True",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)