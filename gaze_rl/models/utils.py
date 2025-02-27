from typing import Dict, List, Tuple

import torch
from omegaconf import DictConfig
from torch import nn

from gaze_rl.utils.logger import log


def make_mlp(
    net_kwargs: DictConfig,
    input_dim: int,
    output_dim: int = None,
    activation: nn.Module = nn.LeakyReLU(0.2),
):
    mlp = []

    prev_dim = input_dim
    for hidden_dim in net_kwargs.hidden_dims:
        mlp.append(nn.Linear(prev_dim, hidden_dim))
        mlp.append(activation)
        prev_dim = hidden_dim

    if output_dim is not None:
        mlp.append(nn.Linear(prev_dim, output_dim))

    output_dim = prev_dim if output_dim is None else output_dim
    mlp = nn.Sequential(*mlp)
    return mlp, output_dim


def make_conv_net(
    input_dim: Tuple,
    net_kwargs: DictConfig,
    output_embedding_dim: int,
    activation: nn.Module = nn.LeakyReLU(0.2),
):
    conv_net = []

    num_conv_layers = len(net_kwargs.hidden_dim)

    # channel must be first
    input_channel = input_dim[0]

    for i in range(num_conv_layers):
        conv_net.append(
            nn.Conv2d(
                input_channel if i == 0 else net_kwargs.hidden_dim[i - 1],
                net_kwargs.hidden_dim[i],
                kernel_size=net_kwargs.kernel_size[i],
                padding=net_kwargs.padding[i],
                stride=net_kwargs.stride[i],
            )
        )
        if i != num_conv_layers - 1:
            conv_net.append(activation)

    conv_net.append(nn.Flatten())

    # compute the output size of the CNN
    with torch.no_grad():
        x = torch.zeros(1, *input_dim)
        for layer in conv_net:
            x = layer(x)
        conv_out_size = x.size(-1)

    log(f"conv_out_size: {conv_out_size}", "yellow")
    conv_net.append(nn.Linear(conv_out_size, output_embedding_dim))

    conv_net = nn.Sequential(*conv_net)
    return conv_net
