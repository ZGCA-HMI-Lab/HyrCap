"""Native one-dimensional deformable attention module."""

import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from .functions.ms_deform_attn_func import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
        )
    return (n & (n - 1) == 0) and n != 0


def deform_attn_core_pytorch(
    value, temporal_lengths, sampling_locations, attention_weights
):
    """Differentiable one-dimensional deformable attention."""
    batch, _, heads, channels = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    lengths = [int(length) for length in temporal_lengths]
    values = value.split(lengths, dim=1)
    grids = 2 * sampling_locations - 1
    grids = torch.cat((grids, torch.zeros_like(grids)), dim=-1)
    sampled = []
    for level, length in enumerate(lengths):
        level_values = (
            values[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch * heads, channels, 1, length)
        )
        level_grid = (
            grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        )
        sampled.append(
            F.grid_sample(
                level_values,
                level_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    weights = attention_weights.transpose(1, 2).reshape(
        batch * heads, 1, queries, levels * points
    )
    output = (
        torch.stack(sampled, dim=-2).flatten(-2) * weights
    ).sum(-1)
    return output.view(batch, heads * channels, queries).transpose(1, 2).contiguous()


class DeformAttn(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_levels=1,
        n_heads=8,
        n_points=4,
        dropout=0.0,
        base_fps=30,
        boundary_aware=False,
    ):
        r"""
        Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads, but got {} and {}".format(
                    d_model, n_heads
                )
            )
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in DeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.boundary_aware = boundary_aware

        self.seq2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.base_fps = base_fps

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        if self.boundary_aware:
            half_points = self.n_points // 2
            grid_init_s = torch.linspace(
                -self.n_points - half_points,
                -self.n_points + half_points,
                self.n_heads * self.n_points // 2 + 1,
            )
            grid_init_e = torch.linspace(
                self.n_points - half_points,
                self.n_points + half_points,
                self.n_heads * self.n_points // 2 + 1,
            )
            grid_init = torch.cat(
                [
                    grid_init_s[: self.n_heads * self.n_points // 4],
                    grid_init_s[self.n_heads * self.n_points // 4 + 1 :],
                    grid_init_e[: self.n_heads * self.n_points // 4],
                    grid_init_e[self.n_heads * self.n_points // 4 + 1 :],
                ],
                dim=0,
            )[:, None]
        else:
            grid_init = torch.linspace(
                -self.n_points, self.n_points, self.n_heads * self.n_points + 1
            )
            grid_init = torch.cat(
                [
                    grid_init[: self.n_heads * self.n_points // 2],
                    grid_init[self.n_heads * self.n_points // 2 + 1 :],
                ],
                dim=0,
            )[:, None]
        grid_init = grid_init.view(self.n_heads, 1, self.n_points, 1).repeat(
            1, self.n_levels, 1, 1
        )

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_temporal_lens,
        input_level_start_index,
        input_padding_mask=None,
        offset_normalizer=None,
    ):
        r"""
        :param query (= src + pos)         (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 1), range in [0, 1], left (0), right (1), including padding area
                                        or (N, Length_{query}, n_levels, 2), add additional (t) to form reference segments
        :param input_flatten (=src)        (N, \sum_{l=0}^{L-1} T_l, C)
        :param input_temporal_lens         (n_levels), [T_0, T_1, ..., T_(L-1)]
        :param input_level_start_index     (n_levels, ), [0, T_0, T_1, T_2, ..., T_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} T_l), True for padding elements, False for non-padding elements
        :param fps                         (N, )
        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert input_temporal_lens.sum() == Len_in

        Len_values = input_temporal_lens[: self.n_levels].sum().item()
        value = self.value_proj(input_flatten[:, :Len_values])
        if input_padding_mask is not None:
            value = value.masked_fill(
                input_padding_mask[:, :Len_values, None], float(0)
            )
        value = value.view(N, Len_values, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 1
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )
        attention_weights = self.dropout(attention_weights)
        if reference_points.shape[-1] == 1:
            if offset_normalizer is None:
                offset_normalizer = input_temporal_lens[
                    None, None, None, : self.n_levels, None, None
                ]
            else:
                offset_normalizer = (
                    offset_normalizer[..., None] / self.base_fps
                ) * input_temporal_lens[None, ...]
                offset_normalizer = offset_normalizer[
                    :, None, None, : self.n_levels, None, None
                ]

            sampling_locations = (
                reference_points[:, :, None, : self.n_levels, None, :]
                + sampling_offsets / offset_normalizer
            )

        elif reference_points.shape[-1] == 2:
            sampling_locations = (
                reference_points[:, :, None, : self.n_levels, None, :1]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, : self.n_levels, None, 1:]
                * 0.5
            )
        input_temporal_lens = input_temporal_lens[: self.n_levels]
        if torch.is_grad_enabled():
            output = deform_attn_core_pytorch(
                value.float(),
                input_temporal_lens,
                sampling_locations.float(),
                attention_weights.float(),
            )
            if value.dtype == torch.float16:
                output = output.to(torch.float16)
                output = self.output_proj(output)
                return output, (sampling_locations, sampling_offsets)
            return output, None

        if value.dtype == torch.float16:
            output = MSDeformAttnFunction.apply(
                value.to(torch.float32),
                input_temporal_lens,
                input_level_start_index,
                sampling_locations.to(torch.float32),
                attention_weights.to(torch.float32),
                self.seq2col_step,
            )
            output = output.to(torch.float16)
            output = self.output_proj(output)
            return output, (sampling_locations, sampling_offsets)

        output = MSDeformAttnFunction.apply(
            value,
            input_temporal_lens,
            input_level_start_index,
            sampling_locations,
            attention_weights,
            self.seq2col_step,
        )
        return output, None
