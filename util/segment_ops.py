"""Temporal segment conversion used by detector post-processing."""

import torch


def segment_cw_to_t1t2(segments):
    """Convert ``(center, width)`` segments to ``(start, end)``."""
    center, width = segments.unbind(-1)
    return torch.stack((center - 0.5 * width, center + 0.5 * width), dim=-1)
