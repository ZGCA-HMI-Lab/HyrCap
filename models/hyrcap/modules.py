"""Neural-network modules used by the HyrCap evaluation path."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MaskedConv1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        if kernel_size % 2 != 1 or kernel_size // 2 != padding:
            raise ValueError("MaskedConv1D requires centered odd-kernel padding")
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, features, mask):
        if features.shape[-1] % self.stride:
            raise ValueError("Feature length must be divisible by convolution stride")
        valid = (~mask).unsqueeze(1)
        output = self.conv(features)
        if self.stride > 1:
            valid = F.interpolate(
                valid.to(features.dtype), size=output.shape[-1], mode="nearest"
            )
        else:
            valid = valid.to(features.dtype)
        output = output * valid.detach()
        return output, ~valid.bool().squeeze(1)


class LayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, features, mask):
        mean = features.mean(dim=1, keepdim=True)
        residual = features - mean
        variance = (residual**2).mean(dim=1, keepdim=True)
        output = residual / torch.sqrt(variance + self.eps)
        return output * self.weight + self.bias, mask


class Sequential(nn.Sequential):
    def forward(self, features, mask):
        for module in self:
            features, mask = module(features, mask)
        return features, mask


class Dropout(nn.Dropout):
    def forward(self, features, mask):
        return super().forward(features), mask


class ReLU(nn.ReLU):
    def forward(self, features, mask):
        return super().forward(features), mask


class MaskedResizer(nn.Module):
    def forward(self, features, mask):
        output_mask = (
            F.interpolate(mask.unsqueeze(1).float(), scale_factor=0.5, mode="linear")
            .squeeze(1)
            .gt(0.5)
        )
        output = torch.zeros(
            features.shape[0],
            features.shape[1],
            output_mask.shape[1],
            device=features.device,
        )
        for index in range(features.shape[0]):
            valid_length = int((~mask[index]).sum())
            resized_length = int((~output_mask[index]).sum())
            resized = F.interpolate(
                features[[index], :, :valid_length],
                size=resized_length,
                mode="linear",
            )
            output[index, :, :resized_length] = resized[0]
        return output, output_mask


class ResizerBackbone(nn.Module):
    def __init__(self, num_feature_levels):
        super().__init__()
        self.resizers = nn.ModuleList(
            MaskedResizer() for _ in range(1, num_feature_levels)
        )

    def forward(self, features, mask):
        features = (~mask).unsqueeze(1).float().detach() * features
        output_features = (features,)
        output_masks = (mask,)
        for resizer in self.resizers:
            features, mask = resizer(features, mask)
            output_features += (features,)
            output_masks += (mask,)
        return output_features, output_masks


class GatedConv(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        num_levels=4,
        hidden_dim=256,
        bias=True,
        group_conv=True,
    ):
        super().__init__()
        if hidden_dim % num_levels:
            raise ValueError("hidden_dim must be divisible by num_levels")
        self.hidden_dim = hidden_dim
        self.conv_dim = hidden_dim // num_levels
        self.convs = nn.ModuleList(
            nn.Conv1d(
                self.conv_dim,
                self.conv_dim,
                kernel_size=kernel_size,
                padding=(level + 1) * (kernel_size // 2),
                dilation=level + 1,
                groups=self.conv_dim if group_conv else 1,
                bias=bias,
            )
            for level in range(num_levels)
        )
        self.fc_in = nn.Linear(dim, hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim, dim)
        self.act = nn.SiLU()

    def forward(self, features, mask=None):
        convolution, gate = torch.split(self.fc_in(features), self.hidden_dim, dim=-1)
        convolution = convolution.transpose(1, 2)
        valid = None
        if mask is not None:
            valid = (~mask).float().unsqueeze(1).detach()
            convolution = convolution * valid
        parts = torch.split(convolution, self.conv_dim, dim=1)
        convolution = torch.cat(
            [layer(part) for layer, part in zip(self.convs, parts, strict=True)],
            dim=1,
        )
        if valid is not None:
            convolution = convolution * valid
        return self.fc_out(convolution.transpose(1, 2) * self.act(gate))


class CPCC(nn.Module):
    """Cross-proposal consensus support and score calibration."""

    def __init__(self, learned=True, in_dim=4, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.consensus_head = (
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            if learned
            else None
        )

    def forward(self, proposals, ratio=1.0, beta=0.15, gate=None):
        if torch.is_tensor(proposals):
            if self.consensus_head is None:
                raise RuntimeError("Tensor inputs require a learned consensus head.")
            return self.consensus_head(proposals).squeeze(-1)
        frame = proposals.copy()
        if gate is not None:
            frame["best_iou_ref"] = frame["best_iou_ref"].astype(float) * np.asarray(
                gate
            )
        return self.rerank(frame, ratio, beta)

    @staticmethod
    def compute_support(segments, references):
        if segments.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if references.size == 0:
            return np.zeros((segments.shape[0],), dtype=np.float32)
        segments = segments.astype(np.float32)
        references = references.astype(np.float32)
        left = np.maximum(segments[:, None, 0], references[None, :, 0])
        right = np.minimum(segments[:, None, 1], references[None, :, 1])
        intersection = np.maximum(right - left, 0.0)
        union = np.maximum(segments[:, None, 1], references[None, :, 1]) - np.minimum(
            segments[:, None, 0], references[None, :, 0]
        )
        overlap = np.where(union > 1e-8, intersection / np.maximum(union, 1e-8), 0.0)
        return overlap.max(axis=1).astype(np.float32)

    @staticmethod
    def rerank(frame, ratio, beta):
        frame = frame.copy()
        frame["score"] = frame["score"].astype(float)
        count = int(round(float(max(0.0, min(1.0, ratio))) * len(frame)))
        frame["score_before_rerank"] = frame["score"].astype(float)
        frame["reranked"] = False
        if count > 0:
            indices = frame["best_iou_ref"].astype(float).nlargest(count).index
            frame.loc[indices, "score"] = frame.loc[indices, "score"].astype(float) * (
                1.0 + float(beta) * frame.loc[indices, "best_iou_ref"].astype(float)
            )
            frame.loc[indices, "reranked"] = True
        return frame


class LQE(nn.Module):
    """Estimate proposal localization quality."""

    def __init__(self, in_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


class PRS(nn.Module):
    """Score proposal redundancy with a prediction-only temporal graph."""

    def __init__(self, in_dim, edge_dim=4, hidden_dim=96, dropout=0.1):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.rank_head = nn.Linear(hidden_dim, 1)
        self.winner_head = nn.Linear(hidden_dim, 1)
        self.dup_head = nn.Linear(hidden_dim, 1)

    def forward(self, features, edge_index=None, edge_features=None):
        hidden = self.node_enc(features)
        if (
            edge_index is not None
            and edge_features is not None
            and edge_index.numel() > 0
        ):
            source = edge_index[0].long()
            destination = edge_index[1].long()
            source_hidden = hidden[source]
            destination_hidden = hidden[destination]
            messages = self.msg_mlp(torch.cat([source_hidden, edge_features], dim=-1))
            attention = torch.sigmoid(
                self.att_mlp(
                    torch.cat(
                        [destination_hidden, source_hidden, edge_features], dim=-1
                    )
                )
            )
            aggregate = torch.zeros_like(hidden)
            aggregate.index_add_(0, destination, messages * attention)
            degree = torch.zeros(
                (hidden.size(0), 1), dtype=hidden.dtype, device=hidden.device
            )
            degree.index_add_(
                0,
                destination,
                torch.ones(
                    (destination.numel(), 1),
                    dtype=hidden.dtype,
                    device=hidden.device,
                ),
            )
            hidden = self.norm(hidden + aggregate / torch.clamp(degree, min=1.0))

        return (
            self.rank_head(hidden).squeeze(-1),
            self.winner_head(hidden).squeeze(-1),
            self.dup_head(hidden).squeeze(-1),
        )

    @staticmethod
    def _temporal_iou(segment, segments):
        intersection = np.maximum(
            np.minimum(segment[1], segments[:, 1])
            - np.maximum(segment[0], segments[:, 0]),
            0.0,
        )
        union = np.maximum(segment[1], segments[:, 1]) - np.minimum(
            segment[0], segments[:, 0]
        )
        return np.where(union > 0, intersection / np.maximum(union, 1e-8), 0.0)

    @staticmethod
    def build_graph(frame, edge_iou=0.5, topk=8, higher_only=True):
        if len(frame) == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        rows = {index: row for row, index in enumerate(frame.index.to_list())}
        sources = []
        destinations = []
        features = []
        for _, group in frame.groupby(["video_id", "pred_label"], sort=False):
            indices = group.index.to_numpy(dtype=np.int64)
            if len(indices) <= 1:
                continue
            segments = group[["t_start", "t_end"]].to_numpy(dtype=np.float32)
            scores = group["detector_score"].to_numpy(dtype=np.float32)
            centers = group["segment_center"].to_numpy(dtype=np.float32)
            lengths = np.maximum(group["segment_len"].to_numpy(dtype=np.float32), 1e-6)
            overlaps = np.zeros((len(indices), len(indices)), dtype=np.float32)
            for index in range(len(indices)):
                overlaps[index] = PRS._temporal_iou(segments[index], segments)
                overlaps[index, index] = 0.0

            for destination in range(len(indices)):
                candidates = np.where(overlaps[destination] >= float(edge_iou))[0]
                if higher_only:
                    candidates = candidates[scores[candidates] >= scores[destination]]
                if candidates.size == 0:
                    continue
                ranking = overlaps[destination, candidates] * (
                    1.0 + np.clip(scores[candidates], 0.0, 1.0)
                )
                neighbors = candidates[np.argsort(-ranking)]
                if topk > 0:
                    neighbors = neighbors[: int(topk)]
                for source in neighbors:
                    sources.append(rows[int(indices[source])])
                    destinations.append(rows[int(indices[destination])])
                    features.append(
                        [
                            float(overlaps[destination, source]),
                            float(scores[source] - scores[destination]),
                            float(
                                abs(centers[source] - centers[destination])
                                / max(
                                    0.5 * (lengths[source] + lengths[destination]),
                                    1e-6,
                                )
                            ),
                            float(
                                np.log(
                                    (lengths[source] + 1e-6)
                                    / (lengths[destination] + 1e-6)
                                )
                            ),
                        ]
                    )

        if not sources:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
        edge_index = np.vstack(
            [
                np.asarray(sources, dtype=np.int64),
                np.asarray(destinations, dtype=np.int64),
            ]
        )
        edge_features = np.asarray(features, dtype=np.float32)
        return edge_index, np.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0)
