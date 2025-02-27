from pathlib import Path
import numpy as np
import tensorflow as tf
from typing import List

def log(message, color=None):
    print(message)

def save_dataset(
    trajectories,
    save_file: Path,
    env_name: str,
    save_imgs: bool,
    framestack: int = 0,
    image_keys: List = [],
):
    log(f"Saving dataset to: {save_file}", "green")
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # create tfds from generator
    def generator():
        for trajectory in trajectories:
            yield trajectory

    # For robosuite, determine shapes from the first trajectory
    first_traj = trajectories[0]
    
    # Determine observation shape
    if isinstance(first_traj["observations"], np.ndarray):
        obs_dim = first_traj["observations"].shape[1] if len(first_traj["observations"].shape) > 1 else 1
    else:
        # Default to a reasonable size if we can't determine
        obs_dim = 20
        log(f"Warning: Could not determine observation dimension, using default {obs_dim}", "yellow")
    
    # Determine action shape
    if isinstance(first_traj["actions"], np.ndarray):
        action_dim = first_traj["actions"].shape[1] if len(first_traj["actions"].shape) > 1 else 1
    else:
        # Default to a reasonable size if we can't determine
        action_dim = 8
        log(f"Warning: Could not determine action dimension, using default {action_dim}", "yellow")
    
    # Create features dict with correct dimensions
    features_dict = {
        "observations": tf.TensorSpec(shape=(None, obs_dim), dtype=np.float32),
        "actions": tf.TensorSpec(shape=(None, action_dim), dtype=np.float32),
        "discount": tf.TensorSpec(shape=(None), dtype=np.float32),
        "rewards": tf.TensorSpec(shape=(None), dtype=np.float32),
        "is_first": tf.TensorSpec(shape=(None,), dtype=np.bool_),
        "is_last": tf.TensorSpec(shape=(None,), dtype=np.bool_),
        "is_terminal": tf.TensorSpec(shape=(None,), dtype=np.bool_),
    }
    
    # Add subgoals if present
    if "subgoals" in first_traj and isinstance(first_traj["subgoals"], np.ndarray):
        subgoal_dim = first_traj["subgoals"].shape[1] if len(first_traj["subgoals"].shape) > 1 else 1
        features_dict["subgoals"] = tf.TensorSpec(shape=(None, subgoal_dim), dtype=np.float32)
    
    # Add image features if needed
    if save_imgs and "images" in trajectories[0]:
        image_features = {}
        for img_key in image_keys:
            if img_key in trajectories[0]["images"] and len(trajectories[0]["images"][img_key]) > 0:
                img = trajectories[0]["images"][img_key][0]
                if isinstance(img, np.ndarray):
                    image_features[img_key] = tf.TensorSpec(shape=(None, *img.shape), dtype=np.uint8)
        
        if image_features:
            features_dict["images"] = image_features
    
    # Create and save the TensorFlow dataset
    log(f"Feature dictionary: {features_dict}")
    trajectory_tfds = tf.data.Dataset.from_generator(
        generator, output_signature=features_dict
    )
    
    tf.data.experimental.save(trajectory_tfds, str(save_file))
    log(f"Dataset saved to {save_file}", "green")