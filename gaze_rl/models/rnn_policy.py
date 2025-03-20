import torch
import torch.nn as nn

# from r3m import load_r3m
import torch.nn.functional as F
from omegaconf import DictConfig
from torchvision.transforms import Resize

from gaze_rl.models.base import BaseModel
from gaze_rl.models.utils import make_mlp
from gaze_rl.utils.logger import log


class RNNPolicy(BaseModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)

        # TODO
        # if gaze_concat and cfg.multi_head:
        #     cfg.input_dim += env_gaze_dim

        self.input_embed = nn.Linear(cfg.input_dim, cfg.input_embed_dim)

        # Batch normalization for input embedding
        self.input_bn = nn.BatchNorm1d(cfg.input_embed_dim)

        self.activation = nn.LeakyReLU(0.2)  # TODO: tune this value (or) make param

        if self.cfg.subgoal_dim:
            self.subgoal_embed = nn.Linear(cfg.subgoal_dim, cfg.subgoal_embed_dim)

            # Batch normalization for subgoal embedding
            self.subgoal_bn = nn.BatchNorm1d(cfg.subgoal_embed_dim)

        rnn_input_dim = cfg.input_embed_dim

        if cfg.subgoal_dim:
            rnn_input_dim += cfg.subgoal_embed_dim

        # TODO
        # if gaze_concat and not cfg.multi_head:
        #     rnn_input_dim += env_gaze_dim

        # RNN layer (LSTM or GRU)
        rnn_class = nn.LSTM if cfg.rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_class(
            input_size=rnn_input_dim,
            hidden_size=cfg.rnn_hidden_dim,
            num_layers=cfg.rnn_num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )

        # Batch norm for RNN output
        self.rnn_bn = nn.BatchNorm1d(
            cfg.rnn_hidden_dim * (2 if cfg.bidirectional else 1)
        )

        # Create hidden layers
        prev_dim = cfg.rnn_hidden_dim * (2 if cfg.bidirectional else 1)

        # embed task label
        # only use task label if we are running bc
        if cfg.input_task:
            self.task_label_embedding = nn.Embedding(
                num_embeddings=self.cfg.num_tasks,
                embedding_dim=self.cfg.task_embedding_dim,
            )
            prev_dim += self.cfg.task_embedding_dim

        self.output_mlp, _ = make_mlp(
            cfg.output_mlp, input_dim=prev_dim, output_dim=cfg.action_dim
        )

    def forward(self, x, subgoal=None, task_label=None, hidden=None):
        is_single_step = x.dim() == 2

        if is_single_step:
            x = x.unsqueeze(1)  # Add time dimension
            if subgoal is not None:
                subgoal = subgoal.unsqueeze(1)

        x = self.input_embed(x)
        x = self.input_bn(x.flatten(0, 1)).view(x.size())

        if subgoal is not None:
            subgoal_embed = self.subgoal_embed(subgoal)
            subgoal_embed = self.subgoal_bn(subgoal_embed.flatten(0, 1)).view(
                subgoal_embed.size()
            )
            x = torch.cat([x, subgoal_embed], dim=-1)

        batch_size, seq_length, _ = x.size()

        rnn_out, hidden = self.rnn(x, hidden)
        rnn_out = self.rnn_bn(rnn_out.flatten(0, 1)).view(rnn_out.size())

        if task_label is not None:
            task_emb = self.task_label_embedding(task_label.long()).view(
                batch_size * seq_length, -1
            )
            rnn_out = torch.cat([rnn_out, task_emb], dim=1)

        rnn_out = self.activation(rnn_out)
        out = self.output_mlp(rnn_out)

        if is_single_step:
            # remove the time dimension
            out = out.squeeze(1)

        return out, hidden
