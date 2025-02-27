import os
from functools import partial
from pathlib import Path

import rlds
import tensorflow as tf
import tensorflow_datasets as tfds
from omegaconf import DictConfig

from gaze_rl.utils.logger import log

os.environ["TFDS_DATA_DIR"] = "/scr/shared/gaze_rl/tensorflow_datasets"


def episode_to_step_custom(episode, size):
    episode = tf.data.Dataset.from_tensor_slices(episode)
    return rlds.transformations.batch(episode, size=size, shift=1, drop_remainder=True)


# add additional fields to the dataset
def add_new_fields(x):
    x["mask"] = tf.ones_like(x["actions"])
    x["timestep"] = tf.range(tf.shape(x["actions"])[0])
    return x


def use_image_observations(x, channel_first: bool = False):
    if "images" in x:
        images = x["images"]

        for key in images:
            if channel_first:
                # has framestack
                has_framestack = len(images[key].shape) == 5

                if has_framestack:
                    import pdb

                    pdb.set_trace()
                else:
                    images[key] = tf.transpose(images[key], perm=[0, 3, 1, 2])

            # if the images are not normalized, normalize them
            if images[key].dtype != tf.float32:
                images[key] = tf.cast(images[key], tf.float32) / 255.0

        x["observations"] = images
    return x


# remove trajectories where the number of steps is less than 2
def filter_fn(traj):
    return tf.math.greater(tf.shape(traj["observations"])[0], 2)


def process_dataset(
    cfg: DictConfig,
    ds: tf.data.Dataset,
    shuffle: bool = True,
    drop_remainder: bool = False,
):
    """
    Applies transformations to base tfds such as batching, shuffling, etc.
    """
    ds = ds.filter(filter_fn)

    # caching the dataset makes it faster in the next iteration
    ds = ds.cache()

    # shuffle the dataset first?
    ds = ds.shuffle(1000, reshuffle_each_iteration=False)

    # limit the number of trajectories that we use
    ds = ds.take(cfg.num_trajs)
    log(f"\ttaking {cfg.num_trajs} trajectories")

    ds = ds.map(add_new_fields)

    # replace observations with images
    if cfg.image_obs:
        log("replace observations with images", "yellow")
        # we want to convert from [H, W, C] to [C, H, W]
        ds = ds.map(partial(use_image_observations, channel_first=True))

    if cfg.data_type == "n_step":
        ds = ds.flat_map(partial(episode_to_step_custom, size=cfg.seq_len))
    elif cfg.data_type == "transitions":
        ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)

    # shuffle the full dataset one more time
    if shuffle:
        log("shuffling dataset")
        ds = ds.shuffle(1000, reshuffle_each_iteration=False)

    if cfg.num_examples != -1:
        log(f"\ttaking {cfg.num_examples} examples")
        ds = ds.take(cfg.num_examples)

        # recommended to do dataset.take(k).cache().repeat()
        ds = ds.cache()

    ds = ds.batch(cfg.batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_dataloader(
    cfg: DictConfig,
    shuffle: bool = True,
):
    """
    Returns a dictionary containing the training and validation datasets.
    Validation dataset is a dictionary of {env_id: dataset}
    """
    datasets = {}
    data_cfg = cfg.data
    # data_dir = Path(data_cfg.data_dir) / "tensorflow_datasets"
    data_dir = Path(data_cfg.data_dir)
    log(f"loading tfds dataset from: {data_dir}")

    env_id = cfg.env.env_id
    ds_name = data_cfg.dataset_name
    log(f"loading dataset for {env_id}")

    save_file = data_dir / ds_name
    ds = tf.data.experimental.load(str(save_file))
    log(f"dataset name: {ds_name}")
    log(f"len of original_ds: {len(ds)}")

    # split dataset into train and eval
    total_trajectories = len(ds)
    num_train = int(total_trajectories * data_cfg.train_frac)
    num_eval = total_trajectories - num_train

    # first shuffle the dataset once
    if shuffle:
        log("shuffling dataset")
        ds = ds.shuffle(1000, reshuffle_each_iteration=False)

    train_ds = ds.take(num_train)
    eval_ds = ds.skip(num_train)
    log(
        f"num train trajs: {num_train}, num eval trajs: {num_eval}, len train: {len(train_ds)}, len eval: {len(eval_ds)}"
    )

    log("processing train dataset")
    train_ds = process_dataset(cfg.data, train_ds, shuffle=shuffle)
    datasets["train"] = {env_id: train_ds}

    # use all the trajectories in the eval dataset
    cfg_eval = cfg.data.copy()
    cfg_eval.num_trajs = -1
    cfg_eval.num_examples = -1
    log("processing eval dataset")
    eval_ds = process_dataset(cfg_eval, eval_ds)

    return train_ds, eval_ds


if __name__ == "__main__":
    pass
