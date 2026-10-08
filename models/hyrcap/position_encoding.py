"""Time-based sinusoidal position encoding."""

import torch
from torch import nn


class TimePositionEncoding(nn.Module):
    def __init__(self, dimension, temperature):
        super().__init__()
        self.dimension = dimension
        self.temperature = temperature

    def forward(self, grids):
        frequencies = torch.arange(
            self.dimension, dtype=torch.float32, device=grids.device
        )
        frequencies = self.temperature ** (
            2 * torch.div(frequencies, 2, rounding_mode="trunc") / self.dimension
        )
        angles = grids[:, :, None, None] / frequencies
        encoding = torch.stack(
            (angles[:, :, :, 0::2].sin(), angles[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)
        return encoding.transpose(2, 1)


def build_position_encoding(config):
    return TimePositionEncoding(config.hidden_dim, config.temperature)
