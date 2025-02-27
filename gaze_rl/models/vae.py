import torch
import torch.nn as nn
from omegaconf import DictConfig

from gaze_rl.models.base import BaseModel
from gaze_rl.models.utils import make_mlp


class Encoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        input_dim = cfg.input_dim

        # if cfg.concat_input_with_gaze and not cfg.multi_head:
        #     # for gaze concatenated approach, changes depending on 2D or 3D gaze
        #     input_dim += cfg.gaze_dim

        self.input_embed = nn.Linear(input_dim, cfg.embed_dim)
        self.cond_embed = nn.Linear(cfg.cond_dim, cfg.embed_dim)

        self.encoder, encoder_output_dim = make_mlp(
            cfg.net, input_dim=cfg.embed_dim * 2
        )
        self.fc_mu = nn.Linear(encoder_output_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, cfg.latent_dim)

    def forward(self, x, cond):
        x = self.input_embed(x)
        cond = self.cond_embed(cond)
        x = torch.cat([x, cond], dim=-1)

        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # clamp logvar
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.input_embed = nn.Linear(cfg.latent_dim, cfg.embed_dim)
        self.cond_embed = nn.Linear(cfg.cond_dim, cfg.embed_dim)

        # if cfg.concat_input_with_gaze and not cfg.multi_head:
        #     # for gaze concatenated approach, changes depending on 2D or 3D gaze
        #     input_dim += cfg.gaze_dim

        self.encoder, encoder_output_dim = make_mlp(
            cfg.net, input_dim=cfg.embed_dim * 2
        )

        if self.cfg.multi_head:
            self.fc_output_joint = nn.Linear(
                encoder_output_dim, cfg.subgoal_dim - 1
            )  # for gripper open in RLbench
            self.fc_output_gripper = nn.Linear(encoder_output_dim, 1)
        else:
            self.fc_output = nn.Linear(encoder_output_dim, cfg.subgoal_dim)

    def forward(self, z, cond):
        z = self.input_embed(z)
        cond = self.cond_embed(cond)
        h = torch.cat([z, cond], dim=-1)

        h = self.encoder(h)

        if self.cfg.multi_head:
            # TODO: generalize this for any multi head output, this is only for RLBench with gripper at the end
            joint_output = self.fc_output_joint(h)
            gripper_output = self.fc_output_gripper(h)
            x_recon = torch.cat([joint_output, gripper_output], dim=-1)
        else:
            x_recon = self.fc_output(h)

        return x_recon


class VAE(BaseModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # if gaze_concat:
        #     cfg.cond_dim += env_gaze_dim
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.latent_dim = cfg.latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond):
        # combine input and cond
        # embed the input
        mu, logvar = self.encoder(x, cond)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, cond)
        return x_recon, mu, logvar
