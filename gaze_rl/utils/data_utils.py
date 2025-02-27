from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    observations: np.ndarray = None
    actions: np.ndarray = None
    rewards: np.ndarray = None
    subgoals: np.ndarray = None
    next_observations: np.ndarray = None
    dones: np.ndarray = None
    tasks: np.ndarray = None
    mask: np.ndarray = None
    timestep: np.ndarray = None
    traj_index: np.ndarray = None
    is_first: np.ndarray = None
    is_last: np.ndarray = None
    is_terminal: np.ndarray = None
    discount: np.ndarray = None
    images: np.ndarray = None
