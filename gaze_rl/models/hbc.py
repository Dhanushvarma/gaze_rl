import einops
import torch
import torch.nn as nn
from omegaconf import DictConfig

from gaze_rl.models.base import BaseModel
from gaze_rl.models.rnn_policy import RNNPolicy
from gaze_rl.models.vae import VAE
from gaze_rl.utils.logger import log


class HierarchicalBC(BaseModel):
    def __init__(self, cfg: DictConfig, input_dim: int = None):
        super().__init__(cfg, input_dim)
        self.hl_policy = VAE(cfg.hl_policy)
        self.ll_policy = RNNPolicy(cfg.ll_policy)

    def forward_hl(self, x, cond):
        return self.hl_policy(x, cond)

    def forward_ll(self, x, subgoal=None, hidden=None):
        return self.ll_policy(x, subgoal=subgoal, hidden=hidden)

    def get_action(self, x, subgoal=None, hidden=None):
        return self.forward_ll(x, subgoal=subgoal, hidden=hidden)

    def _sample_subgoals(self, cond: torch.Tensor, N: int = 100):
        """
        Sample N subgoals from the decoder
        # TODO(dhanush): verify if logic correct, seems fishy.
        """
        # sample from prior first
        latent_sample = torch.randn(N, self.cfg.hl_policy.latent_dim).to(cond.device)

        # repeat cond N times
        cond = cond.repeat(N, 1)
        predicted_subgoal = self.hl_policy.decoder(latent_sample, cond)
        return predicted_subgoal

    def get_subgoal(
        self,
        cond: torch.Tensor,
        N: int = 100,
        subgoal_selection: str = "random",
        goal: torch.Tensor = None,
    ):
        """
        Sample subgoals and select one based on the selection method.
        """
        all_sampled_subgoals = self._sample_subgoals(cond, N=N)

        if subgoal_selection == "random":
            selected_subgoal = all_sampled_subgoals[0]
        elif subgoal_selection == "euclidean":
            # TODO: revisit, assume L2 norm is correct for subgoals
            distances = torch.linalg.norm(all_sampled_subgoals[:, :3] - goal, dim=1)
            selected_indx = torch.argmin(distances)
            selected_subgoal = all_sampled_subgoals[selected_indx]

        selected_subgoal = einops.repeat(selected_subgoal, "D -> B D", B=1)
        return selected_subgoal, all_sampled_subgoals
