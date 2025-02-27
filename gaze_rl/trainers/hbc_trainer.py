from pathlib import Path

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from gaze_rl.models.hbc import HierarchicalBC
from gaze_rl.trainers.offline_trainer import OfflineTrainer
from gaze_rl.utils.logger import log

# FIXME
def compute_joint_gripper_loss(pred, target, pos_weight=1.0, env_name="rlbench"):
    if env_name == "rlbench":
        act_loss = F.mse_loss(pred[:, :-1], target[:, :-1])
        gripper_loss = F.binary_cross_entropy_with_logits(
            pred[:, -1].squeeze(),
            target[:, -1].squeeze(),
            pos_weight=torch.tensor([pos_weight]).to(pred.device),
        )
        act_loss = act_loss.mean()
        gripper_loss = gripper_loss.mean()
    else:
        act_loss = F.mse_loss(pred, target)
        gripper_loss = torch.tensor(0.0).to(pred.device)
    return act_loss, gripper_loss


class HBCTrainer(OfflineTrainer):
    def setup_model(self):
        model = HierarchicalBC(self.cfg.model, input_dim=self.obs_shape)
        return model

    def compute_hl_loss(self, batch, train: bool = True):
        subgoal_recon, mu, logvar = self.model.forward_hl(
            x=batch.subgoals[:, -1], cond=batch.observations[:, 0]
        )
        prior = torch.distributions.Normal(0, 1)
        kl = torch.distributions.kl.kl_divergence(
            torch.distributions.Normal(mu, torch.exp(0.5 * logvar)), prior
        )
        kl_loss = kl.mean()
        kl_loss *= self.cfg.model.hl_policy.kl_weight

        # FIXME
        if self.cfg.env.name == "rlbench" or self.cfg.env.name == "robosuite":
            subgoal_recon[:, -1] = torch.sigmoid(subgoal_recon[:, -1])
            act_loss_hl, gripper_loss_hl = compute_joint_gripper_loss(
                pred=subgoal_recon,
                target=batch.subgoals[:, -1],
                pos_weight=self.cfg.model.hl_policy.bce_gripper_open_weight,
                env_name=self.cfg.env.name,
            )
            gripper_loss_hl = (
                self.cfg.model.hl_policy.gripper_loss_weight * gripper_loss_hl
            )
            recon_loss = act_loss_hl + gripper_loss_hl

        total_loss = recon_loss + kl_loss
        metrics = {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_hl_loss": total_loss.item(),
        }
        return metrics, total_loss

    def compute_ll_loss(self, batch, train: bool = True):
        action_pred, _ = self.model.forward_ll(
            x=batch.observations, subgoal=batch.subgoals
        )

        # flatten B, H dimension for loss computation
        action_pred = einops.rearrange(action_pred, "B H D -> (B H) D")
        gt_actions = einops.rearrange(batch.actions, "B H D -> (B H) D")

        act_loss_ll, gripper_loss_ll = compute_joint_gripper_loss(
            action_pred,
            gt_actions,
            pos_weight=self.cfg.model.ll_policy.bce_gripper_open_weight,
            env_name=self.cfg.env.name,
        )
        gripper_loss_ll *= self.cfg.model.ll_policy.gripper_loss_weight
        total_loss = act_loss_ll + gripper_loss_ll

        metrics = {
            "act_loss": act_loss_ll.item(),
            "gripper_loss": gripper_loss_ll.item(),
            "total_ll_loss": total_loss.item(),
        }
        return metrics, total_loss

    def compute_loss(self, batch, train: bool = True):
        hl_loss_metrics, hl_loss = self.compute_hl_loss(batch)
        ll_loss_metrics, ll_loss = self.compute_ll_loss(batch)
        total_loss = hl_loss + ll_loss
        metrics = {**hl_loss_metrics, **ll_loss_metrics}
        return metrics, total_loss
