import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import autocast

import gaze_rl.utils.general_utils as gutl
import wandb
from gaze_rl.envs.env_utils import make_envs
from gaze_rl.utils.dataloader import get_dataloader
from gaze_rl.utils.general_utils import omegaconf_to_dict
from gaze_rl.utils.logger import log


class BaseTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        if cfg.debug:
            log("RUNNING IN DEBUG MODE", "red")
            # set some default config values
            cfg.num_updates = 10
            cfg.num_evals = 1
            cfg.num_eval_steps = 10
            cfg.env.max_episode_steps = 10

        hydra_cfg = HydraConfig.get()

        # determine if we are sweeping
        launcher = hydra_cfg.runtime["choices"]["hydra/launcher"]
        sweep = launcher in ["local", "slurm"]

        if sweep:
            self.exp_dir = Path(hydra_cfg.sweep.dir) / hydra_cfg.sweep.subdir
        else:
            self.exp_dir = Path(hydra_cfg.run.dir)

        log(f"launcher: {launcher}")
        log(f"experiment dir: {self.exp_dir}")

        # add exp_dir to config
        self.cfg.exp_dir = str(self.exp_dir)

        # set random seeds
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        tf.random.set_seed(cfg.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"using device: {self.device}")

        if self.cfg.mode == "train":
            self.log_dir = self.exp_dir / "logs"
            self.ckpt_dir = self.exp_dir / "model_ckpts"
            self.video_dir = self.exp_dir / "videos"

            # create directories
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.video_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # save config to yaml file
            OmegaConf.save(self.cfg, f=self.exp_dir / "config.yaml")

            wandb_name = self.cfg.wandb_name

            if self.cfg.use_wandb:
                self.wandb_run = wandb.init(
                    # set the wandb project where this run will be logged
                    entity=cfg.wandb_entity,
                    project=cfg.wandb_project,
                    name=wandb_name,
                    notes=self.cfg.wandb_notes,
                    tags=self.cfg.wandb_tags,
                    # track hyperparameters and run metadata
                    config=omegaconf_to_dict(self.cfg),
                    group=self.cfg.group_name,
                )
            else:
                self.wandb_run = None

        if cfg.best_metric == "max":
            self.best_metric = float("-inf")
        else:
            self.best_metric = float("inf")

        log("loading train and eval datasets", "blue")
        # load datasets
        self.train_dataloader, self.eval_dataloader = get_dataloader(cfg)

        # print batch item shapes
        # determine obs_shape based on the dataset
        batch = next(self.train_dataloader.as_numpy_iterator())

        log("=" * 50)
        log("Shapes of batch items:")
        for k, v in batch.items():
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    log(
                        f"{k}/{k1}: {v1.shape}, {v1.dtype}, {v1.min()}, {v1.max()}, {v1.mean()}"
                    )
            else:
                log(f"{k}: {v.shape}, {v.dtype}, {v.min()}, {v.max()}, {v.mean()}")

        observations = batch["observations"]

        if cfg.env.image_obs:
            # NOTE(deprecated): for RLBench images is a dictionary
            # TODO(deprecated): make modifications for Robosuite
            # TODO: now finally need to fix for MuJoCo Env
            if isinstance(observations, dict):
                self.obs_shape = observations[cfg.env.image_keys[0]].shape[1:]
            else:
                if cfg.data.data_type == "n_step":
                    self.obs_shape = observations.shape[2:]
                else:
                    self.obs_shape = observations.shape[1:]
        else:
            self.obs_shape = observations.shape[-1]

        log(f"observation shape: {self.obs_shape}")

        # figure out how many update steps between each validation step
        if self.cfg.eval_every != -1:
            self.eval_every = self.cfg.eval_every
        elif self.cfg.num_evals != -1:
            self.eval_every = int(self.cfg.num_updates // self.cfg.num_evals)
        elif self.cfg.eval_perc != -1:
            self.eval_every = int(self.cfg.num_updates * self.cfg.eval_perc)
        else:
            raise ValueError("no eval interval specified")

        log(f"evaluating model every: {self.eval_every}")

        # initialize env
        log("initializing envs for eval rollouts", "blue")
        self.eval_envs = make_envs(cfg, training=False)

        # initialize model
        self.model = self.setup_model()
        self.model = self.model.to(self.device)
        # self.model = torch.compile(self.model)  # TODO : ??

        # initialize optimizer
        self.optimizer, self.scheduler = self.setup_optimizer_and_scheduler()

        # for mixed precision training
        self.scaler = torch.amp.GradScaler()

        # count number of parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        log("=" * 50)
        log(f"number of parameters: {num_params}")
        log(f"model: {self.model}")

        # count trainable parameters
        num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        log(f"number of trainable parameters: {num_trainable_params}")

        # count frozen/untrainable parameters
        num_frozen_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        log(f"number of frozen parameters: {num_frozen_params}")

    def setup_logging(self):
        pass

    def setup_model(self):
        pass

    def setup_optimizer_and_scheduler(self):
        opt_cls = getattr(torch.optim, self.cfg.optimizer.name)
        optimizer = opt_cls(self.model.parameters(), **self.cfg.optimizer.params)
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfg.lr_scheduler.name)

        log(
            f"using opt: {self.cfg.optimizer.name}, scheduler: {self.cfg.lr_scheduler.name}",
            "yellow",
        )

        # make this a sequential LR scheduler with warmstarts
        warmstart_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=self.cfg.optimizer.num_warmup_steps,
        )

        scheduler = scheduler_cls(optimizer, **self.cfg.lr_scheduler.params)

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [warmstart_scheduler, scheduler],
            milestones=[self.cfg.optimizer.num_warmup_steps],
        )
        return optimizer, scheduler

    def eval(self, step: int):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def save_model(self, ckpt_dict: Dict, metrics: Dict, iter: int = None):
        # use orbax?
        if self.cfg.save_key and self.cfg.save_key in metrics:
            key = self.cfg.save_key
            if (self.cfg.best_metric == "max" and metrics[key] > self.best_metric) or (
                self.cfg.best_metric == "min" and metrics[key] < self.best_metric
            ):
                self.best_metric = metrics[key]
                ckpt_file = self.ckpt_dir / "best.pkl"
                log(
                    f"new best value: {metrics[key]}, saving best model at epoch {iter} to {ckpt_file}"
                )
                with open(ckpt_file, "wb") as f:
                    pickle.dump(ckpt_dict, f)

                # create a file with the best metric in the name, use a placeholder
                best_ckpt_file = self.ckpt_dir / "best.txt"
                with open(best_ckpt_file, "w") as f:
                    f.write(f"{iter}, {metrics[key]}")

        # also save model to ckpt everytime we run evaluation
        ckpt_file = Path(self.ckpt_dir) / f"ckpt_{iter:06d}.pkl"
        log(f"saving checkpoint to {ckpt_file}")
        with open(ckpt_file, "wb") as f:
            torch.save(ckpt_dict, f)

        ckpt_file = Path(self.ckpt_dir) / "latest.pkl"
        with open(ckpt_file, "wb") as f:
            torch.save(ckpt_dict, f)

    def log_to_wandb(self, metrics: Dict, prefix: str = "", step: int = None):
        if self.wandb_run is not None:
            metrics = gutl.prefix_dict_keys(metrics, prefix=prefix)
            self.wandb_run.log(metrics, step=step)

    @property
    def save_dict(self):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        return state_dict
