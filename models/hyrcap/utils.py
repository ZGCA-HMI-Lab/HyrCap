"""Small tensor utilities used by the detector."""

import torch
import torch.nn.functional as F
from torch import nn


def get_feature_grids(mask, fps, window_size, stride):
    batch_size, length = mask.shape
    indices = torch.arange(length, dtype=torch.float32, device=mask.device)
    indices = indices.unsqueeze(0).repeat(batch_size, 1)
    center_frames = indices * stride[:, None] + window_size[:, None] // 2
    return center_frames / fps[:, None]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip(
                [input_dim, *hidden], [*hidden, output_dim], strict=True
            )
        )

    def forward(self, features):
        for index, layer in enumerate(self.layers):
            features = (
                F.relu(layer(features))
                if index < self.num_layers - 1
                else layer(features)
            )
        return features


def _get_activation_fn(name):
    functions = {
        "relu": F.relu,
        "gelu": F.gelu,
        "glu": F.glu,
        "prelu": nn.PReLU(),
        "selu": F.selu,
        "silu": F.silu,
    }
    try:
        return functions[name]
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error
