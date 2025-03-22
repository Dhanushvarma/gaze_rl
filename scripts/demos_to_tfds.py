"""
Convert RLDS-aligned pickle files to a proper RLDS TensorFlow dataset.
Supports both vanilla behavior cloning and hierarchical behavior cloning.
"""

import os
import glob
import pickle
import numpy as np
import tensorflow as tf
import hydra
from omegaconf import DictConfig
from typing import Dict, List, Any, Tuple, Optional

# Constants for RLDS fields
STEPS = "steps"
OBSERVATION = "observation"
ACTION = "action"
REWARD = "reward"
DISCOUNT = "discount"
IS_FIRST = "is_first"
IS_LAST = "is_last" 
IS_TERMINAL = "is_terminal"
SUBGOAL = "subgoal"


def load_episodes_from_directory(directory_path: str) -> List[Dict[str, List[Any]]]:
    """Load episode data from all pickle files in a directory."""
    episodes = []
    pickle_files = sorted(glob.glob(os.path.join(directory_path, "*.pkl")))
    
    if not pickle_files:
        raise ValueError(f"No pickle files found in {directory_path}")
    
    print(f"Found {len(pickle_files)} episodes to process")
    
    for file_path in pickle_files:
        try:
            with open(file_path, "rb") as f:
                episode_data = pickle.load(f)
                episodes.append(episode_data)
            print(f"Loaded episode from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return episodes


def process_observation(observation: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Extract selected keys from an observation dictionary."""
    return {key: observation[key] for key in keys if key in observation}


def process_action(action_tuple: Tuple) -> Dict[str, tf.Tensor]:
    """Process tuple action (Box, Discrete) into a structured format."""
    if isinstance(action_tuple, tuple) and len(action_tuple) == 2:
        continuous_action, discrete_action = action_tuple
        
        # Ensure numpy arrays
        if not isinstance(continuous_action, np.ndarray):
            continuous_action = np.array(continuous_action, dtype=np.float32)
        
        if not isinstance(discrete_action, np.ndarray) and not np.isscalar(discrete_action):
            discrete_action = np.array([discrete_action], dtype=np.int32)
        elif np.isscalar(discrete_action):
            discrete_action = np.array([discrete_action], dtype=np.int32)
        
        return {
            "continuous": tf.convert_to_tensor(continuous_action, dtype=tf.float32),
            "discrete": tf.convert_to_tensor(discrete_action, dtype=tf.int32)
        }
    else:
        return {"action": tf.convert_to_tensor(action_tuple)}


def get_future_observation(observations: List[Dict[str, Any]], current_idx: int, 
                         horizon: int, keys: List[str]) -> Dict[str, Any]:
    """
    Get a future observation at specified horizon as a subgoal.
    Uses the last observation if the requested horizon exceeds episode length.
    """
    future_idx = current_idx + horizon
    
    # If the requested future index exceeds the episode length,
    # use the last observation instead
    if future_idx >= len(observations):
        future_idx = len(observations) - 1
        
    return process_observation(observations[future_idx], keys)


def episode_to_rlds_format(episode: Dict[str, List[Any]], config: DictConfig) -> tf.data.Dataset:
    """Convert an episode to RLDS format with steps."""
    # Extract observations and actions
    observations = episode["observations"]
    actions = episode["actions"]
    rewards = episode["rewards"]
    discounts = episode["discounts"]
    is_first = episode["is_first"]
    is_last = episode["is_last"]
    is_terminal = episode["is_terminal"]
    
    episode_length = len(observations)
    steps_data = []
    
    obs_keys = config.observation_keys
    
    # Process each step
    for i in range(episode_length):
        # Process observation
        obs = process_observation(observations[i], obs_keys)
        
        # Get subgoal if hierarchical
        subgoal = None
        if config.hierarchical and i < episode_length - 1:  # Skip last step for subgoal
            subgoal = get_future_observation(
                observations, i, config.goal_horizon, 
                config.subgoal_keys
            )
        
        # Create step data
        step_data = {
            OBSERVATION: obs,
            ACTION: process_action(actions[i]),
            REWARD: tf.convert_to_tensor(rewards[i], dtype=tf.float32),
            DISCOUNT: tf.convert_to_tensor(discounts[i], dtype=tf.float32),
            IS_TERMINAL: tf.convert_to_tensor(is_terminal[i], dtype=tf.bool),
            IS_FIRST: tf.convert_to_tensor(is_first[i], dtype=tf.bool),
            IS_LAST: tf.convert_to_tensor(is_last[i], dtype=tf.bool),
        }
        
        # Add subgoal if available
        if subgoal is not None:
            step_data[SUBGOAL] = subgoal
            
        steps_data.append(step_data)
    
    # Convert to TF dataset
    steps_dataset = tf.data.Dataset.from_tensor_slices(steps_data)
    
    return steps_dataset


def create_rlds_dataset(episodes: List[Dict[str, List[Any]]], config: DictConfig) -> tf.data.Dataset:
    """Create a RLDS dataset from episodes."""
    # Convert episodes to RLDS format
    rlds_episodes = []
    
    for episode in episodes:
        steps_dataset = episode_to_rlds_format(episode, config)
        rlds_episodes.append({STEPS: steps_dataset})
    
    # Create dataset of episodes
    return tf.data.Dataset.from_tensor_slices(rlds_episodes)


def create_vae_training_data(episodes: List[Dict[str, List[Any]]], config: DictConfig) -> List[Dict[str, Any]]:
    """Create training examples for VAE subgoal predictor."""
    vae_examples = []
    
    for episode in episodes:
        observations = episode["observations"]
        episode_length = len(observations)
        
        for i in range(episode_length - config.goal_horizon):
            # Current state
            current_state = process_observation(observations[i], config.observation_keys)
            
            # Future state (subgoal)
            future_state = process_observation(
                observations[i + config.goal_horizon], 
                config.subgoal_keys
            )
            
            # Add the pair
            vae_examples.append({
                "state": current_state,
                "goal": future_state
            })
    
    return vae_examples


def create_policy_training_data(episodes: List[Dict[str, List[Any]]], config: DictConfig) -> List[Dict[str, Any]]:
    """Create training examples for policy networks."""
    policy_examples = []
    
    for episode in episodes:
        observations = episode["observations"]
        actions = episode["actions"]
        episode_length = len(observations)
        
        for i in range(episode_length - config.skill_horizon):
            # Skip if we can't get a complete action sequence
            if i + config.skill_horizon >= episode_length:
                continue
                
            # Current state
            current_state = process_observation(observations[i], config.observation_keys)
            
            # Action sequence
            action_sequence = [process_action(actions[i + j]) for j in range(config.skill_horizon)]
            
            example = {
                "state": current_state,
                "actions": action_sequence
            }
            
            # Add goal state if hierarchical
            if config.hierarchical and i + config.goal_horizon < episode_length:
                goal_state = process_observation(
                    observations[i + config.goal_horizon], 
                    config.subgoal_keys
                )
                example["goal"] = goal_state
            
            policy_examples.append(example)
    
    return policy_examples


def save_vae_policy_data(vae_data, policy_data, config):
    """Save VAE and policy training data to disk."""
    if vae_data and config.generate_vae_data:
        with open(config.vae_data_path, 'wb') as f:
            pickle.dump(vae_data, f)
        print(f"VAE training data saved to {config.vae_data_path}")
        
    if policy_data and config.generate_policy_data:
        with open(config.policy_data_path, 'wb') as f:
            pickle.dump(policy_data, f)
        print(f"Policy training data saved to {config.policy_data_path}")


def save_rlds_dataset(dataset: tf.data.Dataset, output_path: str) -> None:
    """Save RLDS dataset to disk in TFRecord format."""
    os.makedirs(output_path, exist_ok=True)
    
    # Save the dataset as TFRecord files
    tf.data.experimental.save(
        dataset,
        output_path,
        compression="GZIP"
    )
    
    print(f"Dataset saved to {output_path}")


@hydra.main(config_path="./config")
def main(cfg: DictConfig):
    print(f"Configuration: \n{cfg}")
    
    # Load episodes
    print(f"Loading episodes from {cfg.data_dir}...")
    episodes = load_episodes_from_directory(cfg.data_dir)
    print(f"Loaded {len(episodes)} episodes")
    
    # Create RLDS dataset
    print("Converting episodes to RLDS format...")
    rlds_dataset = create_rlds_dataset(episodes, cfg)
    
    # Save RLDS dataset
    save_rlds_dataset(rlds_dataset, cfg.output_dir)
    
    # Generate VAE training data if requested
    vae_data = None
    if cfg.hierarchical and cfg.generate_vae_data:
        print("Generating VAE training data...")
        vae_data = create_vae_training_data(episodes, cfg)
    
    # Generate policy training data if requested
    policy_data = None
    if cfg.generate_policy_data:
        print("Generating policy training data...")
        policy_data = create_policy_training_data(episodes, cfg)
    
    # Save additional data
    save_vae_policy_data(vae_data, policy_data, cfg)
    
    print("Conversion complete!")


if __name__ == "__main__":
    main()