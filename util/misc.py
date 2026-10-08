"""Padded temporal tensor container used by HyrCap inference."""

import torch


class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        return NestedTensor(self.tensors.to(device), self.mask.to(device))


def nested_tensor_from_tensor_list(tensors):
    """Pad ``[C, T]`` tensors to a shared length divisible by 32."""
    if not tensors or any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("Expected a non-empty list of [channels, time] tensors")
    channels = tensors[0].shape[0]
    if any(tensor.shape[0] != channels for tensor in tensors):
        raise ValueError("All tensors must have the same channel count")
    maximum = max(tensor.shape[1] for tensor in tensors)
    padded_length = ((maximum + 31) // 32) * 32
    batch = torch.zeros(
        len(tensors),
        channels,
        padded_length,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )
    mask = torch.ones(
        len(tensors), padded_length, dtype=torch.bool, device=tensors[0].device
    )
    for source, destination, item_mask in zip(tensors, batch, mask, strict=True):
        destination[:, : source.shape[1]].copy_(source)
        item_mask[: source.shape[1]] = False
    return NestedTensor(batch, mask)
