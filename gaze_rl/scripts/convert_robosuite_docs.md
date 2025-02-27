# Robosuite HDF5 to TFDS Conversion Script: Options Overview

## Required Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--data_dir` | string | `/path/to/robosuite/data` | Path to directory containing the HDF5 dataset file |
| `--dataset_file` | string | `dataset.hdf5` | Name of the HDF5 dataset file to convert |

## Output Configuration
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--env_name` | string | `robosuite` | Environment name for the output directory structure |
| `--task` | string | Empty (uses dataset filename) | Task name for the output subdirectory |

## Data Selection
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--filter_key` | string | Empty (uses all demos) | Filter key from mask group (e.g., `train`, `valid`) |

## Content Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--save_imgs` | flag | False | Whether to include images in the converted dataset |
| `--compute_subgoals` | flag | False | Whether to extract and save subgoals |
| `--obs_keys` | list | `['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']` | Observation keys to include |
| `--subgoal_keys` | list | `['robot0_eef_pos', 'robot0_gripper_qpos']` | Keys to use for subgoal computation |
| `--image_keys` | list | `['agentview_image']` | Image keys to include when `save_imgs` is True |

## Example Usage

Basic conversion with default options:
```
python convert_robosuite_to_tfds.py --data_dir /path/to/data --dataset_file demo.hdf5
```

Including images:
```
python convert_robosuite_to_tfds.py --data_dir /path/to/data --dataset_file demo.hdf5 --save_imgs
```

Using specific observation keys and computing subgoals:
```
python convert_robosuite_to_tfds.py --data_dir /path/to/data --dataset_file demo.hdf5 --compute_subgoals --obs_keys robot0_eef_pos object_pos robot0_gripper_qpos
```