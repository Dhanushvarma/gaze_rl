import collections
import time
import types
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import tqdm
from omegaconf import DictConfig
from rich.pretty import pretty_repr

import gaze_rl.utils.general_utils as gutl
from gaze_rl.trainers.base_trainer import BaseTrainer
from gaze_rl.utils.data_utils import Batch
from gaze_rl.utils.logger import log
from gaze_rl.utils.rollouts import run_eval_rollouts


class OfflineTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.train_step = 0

    def train(self):
        # first eval
        if not self.cfg.skip_first_eval:
            self.eval(step=0)

        self.model.train()

        if isinstance(self.train_dataloader, types.GeneratorType):
            train_iter = self.train_dataloader
        else:
            train_iter = self.train_dataloader.repeat().as_numpy_iterator()

        for self.train_step in tqdm.tqdm(
            range(self.cfg.num_updates),
            desc=f"{self.cfg.name} train batches",
            disable=False,
            total=self.cfg.num_updates,
        ):
            batch_load_time = time.time()
            batch = next(train_iter)
            # put the batch on the device
            batch = gutl.to_device(batch, self.device)
            batch_load_time = time.time() - batch_load_time
            batch = Batch(**batch)

            # perform a single gradient step
            update_time = time.time()

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                metrics, total_loss = self.compute_loss(batch, train=True)

            self.scaler.scale(total_loss).backward()

            # Unscale gradients to prepare for gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.cfg.clip_grad_norm
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # step the scheduler at the end of everything
            self.scheduler.step()
            metrics["time/batch_load"] = batch_load_time
            metrics["time/update"] = time.time() - update_time

            # get lr
            metrics["lr"] = self.scheduler.get_last_lr()[0]
            self.log_to_wandb(metrics, prefix="train/")

            # log stats about the model params
            param_stats = defaultdict(float)
            for name, param in self.model.named_parameters():
                param_stats[f"{name}_mean"] = param.mean().item()
                param_stats[f"{name}_std"] = param.std().item()

            self.log_to_wandb(param_stats, prefix="params/")

            # log a step counter for wandb
            self.log_to_wandb({"_update": self.train_step}, prefix="step/")

            # run evaluation for each evaluation environment
            if ((self.train_step + 1) % self.eval_every) == 0:
                self.eval(step=self.train_step + 1)

            # log to terminal
            if ((self.train_step + 1) % self.cfg.log_terminal_every) == 0:
                log(f"step: {self.train_step}, train:")
                log(f"{pretty_repr(metrics)}")

        # final evaluation
        self.eval(step=self.cfg.num_updates)

        if self.wandb_run is not None:
            self.wandb_run.finish()

    def eval(self, step: int):
        log("running evaluation", "blue")

        self.model.eval()

        eval_time = time.time()
        eval_iter = self.eval_dataloader.as_numpy_iterator()

        eval_metrics = collections.defaultdict(list)
        for batch in tqdm.tqdm(
            eval_iter,
            desc=f"{self.cfg.name} eval batches",
        ):
            # put the batch on the device
            batch = gutl.to_device(batch, self.device)
            batch = Batch(**batch)

            with torch.no_grad():
                metrics, total_eval_loss = self.compute_loss(batch, train=False)

            for k, v in metrics.items():
                eval_metrics[k].append(v)

        # average metrics over all eval batches
        for k, v in eval_metrics.items():
            eval_metrics[k] = np.mean(np.array(v))

        eval_metrics["time"] = time.time() - eval_time

        self.log_to_wandb(eval_metrics, prefix="eval/")

        # write evaluation metrics to log file
        with open(self.log_dir / "eval.txt", "a+") as f:
            f.write(f"{step}, {eval_metrics}\n")

        log(f"eval: {pretty_repr(eval_metrics)}")

        # run evaluation rollouts
        if self.cfg.run_eval_rollouts:
            rollout_metrics, *_ = run_eval_rollouts(
                cfg=self.cfg,
                envs=self.eval_envs,
                model=self.model,
                wandb_run=self.wandb_run,
                device=self.device,
            )
            self.log_to_wandb(rollout_metrics, prefix="eval_rollout/")

            with open(self.log_dir / "eval.txt", "a+") as f:
                f.write(f"{step}, {rollout_metrics}\n")

            log(f"eval rollout: {pretty_repr(rollout_metrics)}")

        # also save model here
        self.save_model(ckpt_dict=self.save_dict, metrics=eval_metrics, iter=step)

        # set back to train mode
        self.model.train()
        return eval_metrics
