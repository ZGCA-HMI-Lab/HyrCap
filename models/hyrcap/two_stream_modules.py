"""Two-stream projection used by the released detector."""

import torch
from torch import nn

from .modules import Dropout, LayerNorm, MaskedConv1D, ReLU, Sequential


class SingleStreamBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.embd = nn.ModuleList(
            [
                Sequential(
                    MaskedConv1D(input_dim, hidden_dim, 3, padding=1, bias=False),
                    LayerNorm(hidden_dim),
                    Dropout(dropout),
                )
            ]
        )
        self.norm = LayerNorm(hidden_dim)

    def forward(self, features, mask):
        features = features[:, : self.input_dim]
        for layer in self.embd:
            features, mask = layer(features, mask)
            features = features * (~mask.unsqueeze(1)).detach()
        return (features,), (mask,)


class StreamProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.proj = Sequential(
            MaskedConv1D(input_dim, hidden_dim, kernel_size=1, padding=0, bias=False),
            LayerNorm(hidden_dim),
            Dropout(dropout),
        )
        self.refine = Sequential(
            MaskedConv1D(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            LayerNorm(hidden_dim),
            ReLU(inplace=True),
        )

    def forward(self, features, mask):
        features, mask = self.proj(features, mask)
        return self.refine(features, mask)


class DualStreamProjConcatBackbone(nn.Module):
    """Project both streams, concatenate them, then compress to model width."""

    def __init__(
        self, hidden_dim, modality_dims, dropout=0.0, use_concat_residual=True
    ):
        super().__init__()
        if len(modality_dims) != 2 or min(modality_dims) <= 0:
            raise ValueError("modality_dims must contain two positive channel counts")
        self.modality_dims = tuple(modality_dims)
        self.use_concat_residual = use_concat_residual
        self.intern_proj = StreamProjection(modality_dims[0], hidden_dim, dropout)
        self.vjepa_proj = StreamProjection(modality_dims[1], hidden_dim, dropout)
        self.fuse = Sequential(
            MaskedConv1D(
                hidden_dim * 2,
                hidden_dim,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            LayerNorm(hidden_dim),
            ReLU(inplace=True),
        )

    def forward(self, features, mask):
        if features.shape[1] != sum(self.modality_dims):
            raise ValueError(
                f"Expected {sum(self.modality_dims)} channels, got {features.shape[1]}"
            )
        intern, vjepa = torch.split(features, self.modality_dims, dim=1)
        intern, intern_mask = self.intern_proj(intern, mask)
        vjepa, vjepa_mask = self.vjepa_proj(vjepa, mask)
        if not torch.equal(intern_mask, vjepa_mask):
            raise ValueError("Feature stream masks are not aligned")
        fused, output_mask = self.fuse(torch.cat((intern, vjepa), dim=1), intern_mask)
        output = fused + intern if self.use_concat_residual else fused
        output = output * (~output_mask).unsqueeze(1).to(output.dtype)
        return (output,), (output_mask,)
