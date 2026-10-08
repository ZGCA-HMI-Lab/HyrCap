"""Class-agnostic hard NMS for temporal segments."""

import numpy as np
import torch

try:
    from . import nms_1d_cpu
except ImportError:
    import nms_1d_cpu


def apply_nms(detections, nms_threshold, minimum_score, top_k):
    """Filter ``[start, end, score, label]`` rows with hard temporal NMS."""
    segments = torch.from_numpy(detections[:, :2]).float()
    scores = torch.from_numpy(detections[:, 2]).float()
    labels = torch.from_numpy(detections[:, 3]).float()

    valid = scores > minimum_score
    segments = segments[valid]
    scores = scores[valid]
    labels = labels[valid]
    indices = nms_1d_cpu.nms(
        segments.contiguous(), scores.contiguous(), iou_threshold=float(nms_threshold)
    )
    if top_k > 0:
        indices = indices[:top_k]
    return np.column_stack(
        (
            segments[indices].numpy(),
            scores[indices].numpy(),
            labels[indices].numpy(),
        )
    )


def box_voting(segments, scores, iou_threshold, score_offset):
    order = np.argsort(-scores)
    used = np.zeros(len(order), dtype=bool)
    output_segments = []
    output_scores = []
    for position, index in enumerate(order):
        if used[position]:
            continue
        intersection = np.maximum(
            np.minimum(segments[index, 1], segments[order, 1])
            - np.maximum(segments[index, 0], segments[order, 0]),
            0.0,
        )
        union = (
            np.maximum(segments[index, 1], segments[order, 1])
            - np.minimum(segments[index, 0], segments[order, 0])
        )
        overlap = np.where(union > 0, intersection / np.maximum(union, 1e-8), 0.0)
        cluster = np.flatnonzero((overlap >= iou_threshold) & ~used)
        used[cluster] = True
        selected = order[cluster]
        weights = np.maximum(scores[selected] - score_offset, 0.0)
        if float(weights.sum()) <= 0:
            weights = np.ones_like(weights)
        output_segments.append(
            [
                float((weights * segments[selected, 0]).sum() / weights.sum()),
                float((weights * segments[selected, 1]).sum() / weights.sum()),
            ]
        )
        output_scores.append(float(scores[selected].max()))
    return (
        np.asarray(output_segments, np.float64).reshape(-1, 2),
        np.asarray(output_scores, np.float64),
    )
