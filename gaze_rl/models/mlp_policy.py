from typing import Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from omegaconf import DictConfig

from gaze_rl.models.base import BaseModel
from gaze_rl.models.utils import make_conv_net, make_mlp
from gaze_rl.utils.logger import log


class MLPPolicy(BaseModel):
    def __init__(
        self,
        cfg: DictConfig,
        input_dim: Union[int, tuple],
        output_dim: int,
        feature_extractor: nn.Module = None,
    ):
        super(MLPPolicy, self).__init__(cfg, input_dim)
        self.name = "MLPPolicy"

        if feature_extractor is not None:
            # we use a pre-trained feature extractor
            self.input_embedding = feature_extractor

            # get the feature dimension
            with torch.no_grad():
                dummy_input = torch.zeros((1, 3, 224, 224)).cuda()
                embedding_dim = feature_extractor(dummy_input).shape[-1]
        else:
            # encode the input
            if cfg.image_obs:
                self.input_embedding = make_conv_net(
                    input_dim,
                    net_kwargs=cfg.encoder,
                    output_embedding_dim=cfg.embedding_dim,
                )
            else:
                self.input_embedding = nn.Linear(input_dim, cfg.embedding_dim)

            embedding_dim = cfg.embedding_dim

        # policy network
        self.policy, _ = make_mlp(
            input_dim=embedding_dim,
            net_kwargs=cfg.policy,
            output_dim=output_dim,
        )

        if cfg.gaussian_policy:
            self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        if not self.cfg.finetune_feature_extractor:
            with torch.no_grad():
                input_embed = self.input_embedding(x)
        else:
            input_embed = self.input_embedding(x)

        if self.cfg.gaussian_policy:
            mean = self.policy(input_embed)
            std = torch.exp(self.log_std)
            return mean, std

        return self.policy(input_embed)

    def get_action(self, x, **kwargs):
        return self.forward(x)
