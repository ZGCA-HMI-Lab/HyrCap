"""Load the two THUMOS14 feature streams and detector-training targets."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


INTERN_DIMENSION = 3200
VJEPA_TOKEN_DIMENSION = 1408
VJEPA_SPATIAL_SIZE = 16
VJEPA_SPATIAL_TOKENS = VJEPA_SPATIAL_SIZE**2
VJEPA_TEMPORAL_TOKENS_PER_CLIP = 32
VJEPA_TOKENS_PER_CLIP = VJEPA_TEMPORAL_TOKENS_PER_CLIP * VJEPA_SPATIAL_TOKENS
VJEPA_FEATURE_DIMENSION = VJEPA_TOKEN_DIMENSION * 6
FEATURE_DIMENSIONS = (INTERN_DIMENSION, VJEPA_FEATURE_DIMENSION)


def _pool_vjepa_grid(grid, source):
    """Convert ``[T, 16, 16, 1408]`` tokens to the saved 8448-D layout."""
    if grid.shape[0] == 0:
        raise ValueError(f"{source}: V-JEPA token input is empty")
    if grid.shape[1:] != (
        VJEPA_SPATIAL_SIZE,
        VJEPA_SPATIAL_SIZE,
        VJEPA_TOKEN_DIMENSION,
    ):
        raise ValueError(
            f"{source}: expected V-JEPA grid [T, 16, 16, 1408], got {grid.shape}"
        )
    half = VJEPA_SPATIAL_SIZE // 2
    pooled_chunks = []
    for start in range(0, grid.shape[0], VJEPA_TEMPORAL_TOKENS_PER_CLIP):
        chunk = grid[start : start + VJEPA_TEMPORAL_TOKENS_PER_CLIP]
        if not np.isfinite(chunk).all():
            raise ValueError(f"{source}: V-JEPA tokens contain non-finite values")
        pieces = [
            chunk.mean(axis=(1, 2), dtype=np.float32),
            chunk.max(axis=(1, 2)).astype(np.float32, copy=False),
        ]
        pieces.extend(
            chunk[:, y : y + half, x : x + half].mean(
                axis=(1, 2), dtype=np.float32
            )
            for y in (0, half)
            for x in (0, half)
        )
        pooled_chunks.append(np.concatenate(pieces, axis=1))
    return np.concatenate(pooled_chunks, axis=0)


def prepare_vjepa_features(array, source="V-JEPA feature array"):
    """Validate or convert V-JEPA features to ``[T, 8448]``.

    Accepted raw layouts are model tokens grouped as ``[N, 8192, 1408]``,
    spatial tokens as ``[T, 256, 1408]``, or their explicit 16x16-grid forms.
    Already pooled features are returned unchanged. No source file is rewritten.
    """
    array = np.asarray(array)

    if array.ndim == 2 and array.shape[1] == VJEPA_FEATURE_DIMENSION:
        pooled = array
    elif (
        array.ndim == 2
        and array.shape[0] == VJEPA_FEATURE_DIMENSION
        and array.shape[1] != VJEPA_FEATURE_DIMENSION
    ):
        pooled = array.T
    else:
        grid = None
        if (
            array.ndim == 2
            and array.shape[1] == VJEPA_TOKEN_DIMENSION
            and array.shape[0] % VJEPA_TOKENS_PER_CLIP == 0
        ):
            grid = array.reshape(
                -1,
                VJEPA_SPATIAL_SIZE,
                VJEPA_SPATIAL_SIZE,
                VJEPA_TOKEN_DIMENSION,
            )
        elif array.ndim == 3 and array.shape[1:] == (
            VJEPA_SPATIAL_TOKENS,
            VJEPA_TOKEN_DIMENSION,
        ):
            grid = array.reshape(
                -1,
                VJEPA_SPATIAL_SIZE,
                VJEPA_SPATIAL_SIZE,
                VJEPA_TOKEN_DIMENSION,
            )
        elif array.ndim == 3 and array.shape[1:] == (
            VJEPA_TOKENS_PER_CLIP,
            VJEPA_TOKEN_DIMENSION,
        ):
            grid = array.reshape(
                -1,
                VJEPA_SPATIAL_SIZE,
                VJEPA_SPATIAL_SIZE,
                VJEPA_TOKEN_DIMENSION,
            )
        elif array.ndim == 4 and array.shape[1:] == (
            VJEPA_SPATIAL_SIZE,
            VJEPA_SPATIAL_SIZE,
            VJEPA_TOKEN_DIMENSION,
        ):
            grid = array
        elif array.ndim == 4 and array.shape[1:] == (
            VJEPA_TEMPORAL_TOKENS_PER_CLIP,
            VJEPA_SPATIAL_TOKENS,
            VJEPA_TOKEN_DIMENSION,
        ):
            grid = array.reshape(
                -1,
                VJEPA_SPATIAL_SIZE,
                VJEPA_SPATIAL_SIZE,
                VJEPA_TOKEN_DIMENSION,
            )
        elif array.ndim == 5 and array.shape[1:] == (
            VJEPA_TEMPORAL_TOKENS_PER_CLIP,
            VJEPA_SPATIAL_SIZE,
            VJEPA_SPATIAL_SIZE,
            VJEPA_TOKEN_DIMENSION,
        ):
            grid = array.reshape(
                -1,
                VJEPA_SPATIAL_SIZE,
                VJEPA_SPATIAL_SIZE,
                VJEPA_TOKEN_DIMENSION,
            )

        if grid is None:
            raise ValueError(
                f"{source}: expected pooled [T, 8448] features or raw V-JEPA "
                "tokens with 1408 channels and a 16x16 spatial grid; "
                f"got {array.shape}"
            )
        pooled = _pool_vjepa_grid(grid, source)

    if pooled.shape[0] == 0 or pooled.shape[1] != VJEPA_FEATURE_DIMENSION:
        raise ValueError(
            f"{source}: expected processed V-JEPA shape [T, 8448], got {pooled.shape}"
        )
    if not np.isfinite(pooled).all():
        raise ValueError(f"{source}: V-JEPA features contain non-finite values")
    return pooled


def load_features(video, intern_folder, vjepa_folder):
    """Return aligned InternVideo2 and V-JEPA2 features as ``[11648, T]``."""
    intern_path = Path(intern_folder) / f"{video}.npy"
    intern_array = np.load(intern_path, allow_pickle=False)
    if (
        intern_array.ndim != 2
        or intern_array.shape[0] == 0
        or intern_array.shape[1] != INTERN_DIMENSION
    ):
        raise ValueError(
            f"{intern_path}: expected [T, {INTERN_DIMENSION}], got {intern_array.shape}"
        )
    if not np.isfinite(intern_array).all():
        raise ValueError(f"{intern_path}: features contain non-finite values")

    vjepa_path = Path(vjepa_folder) / f"{video}.npy"
    vjepa_array = prepare_vjepa_features(
        np.load(vjepa_path, allow_pickle=False, mmap_mode="r"),
        source=str(vjepa_path),
    )

    intern = torch.from_numpy(np.ascontiguousarray(intern_array.T)).float()
    vjepa = torch.from_numpy(np.ascontiguousarray(vjepa_array.T)).float()
    if vjepa.shape[1] != intern.shape[1]:
        vjepa = F.interpolate(
            vjepa[None], size=intern.shape[1], mode="linear", align_corners=False
        )[0]
    return torch.cat((intern, vjepa), dim=0)


def _crop_training_features(features, segments, labels, config, rng):
    """Crop long sequences while keeping at least one annotated action."""
    length = features.shape[1]
    crop_length = min(length, config["max_seq_len"])
    if length <= config["max_seq_len"]:
        crop_length = rng.randint(max(round(0.9 * length), 1), length)
    if crop_length == length:
        return features, segments, labels

    fps = config["default_fps"]
    stride = config["stride"]
    duration = ((crop_length - 1) * stride + config["base_frame"]) / fps
    action_lengths = (segments[:, 1] - segments[:, 0]).clamp_min(1e-6)
    for _ in range(200):
        start_index = rng.randint(0, length - crop_length)
        start = start_index * stride / fps
        intersection = (
            torch.minimum(segments[:, 1], torch.tensor(start + duration))
            - torch.maximum(segments[:, 0], torch.tensor(start))
        ).clamp_min(0)
        valid = intersection / action_lengths > 0.3
        if valid.any():
            break
    else:
        raise ValueError("No annotated action remains after temporal cropping")

    return (
        features[:, start_index : start_index + crop_length],
        (segments[valid] - start).clamp(0, duration),
        labels[valid],
    )


class THUMOS14TrainingDataset(Dataset):
    """Deterministic crops and targets for the 200-video validation split."""

    def __init__(self, database, intern_folder, vjepa_folder, config):
        self.database = database
        self.intern_folder = Path(intern_folder)
        self.vjepa_folder = Path(vjepa_folder)
        self.config = config
        self.videos = sorted(
            video for video, item in database.items() if item["subset"] == "val"
        )
        if len(self.videos) != 200:
            raise ValueError(f"Expected 200 THUMOS14 val videos, got {len(self.videos)}")
        classes = sorted(
            {
                annotation["label"]
                for video in self.videos
                for annotation in database[video].get("annotations", [])
                if annotation.get("label") != "Ambiguous"
            }
        )
        if len(classes) != config["num_classes"]:
            raise ValueError(
                f"Expected {config['num_classes']} action classes, got {len(classes)}"
            )
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.epoch = 0
        for folder in (self.intern_folder, self.vjepa_folder):
            missing = [
                video for video in self.videos if not (folder / f"{video}.npy").is_file()
            ]
            if missing:
                raise FileNotFoundError(f"{folder}: missing {missing[0]}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        metadata = self.database[video]
        features = load_features(video, self.intern_folder, self.vjepa_folder)
        annotations = []
        for annotation in metadata["annotations"]:
            start, end = annotation["segment"]
            if annotation.get("label") == "Ambiguous" or end - start < 0.02:
                continue
            duplicate = any(
                saved["label"] == annotation["label"]
                and abs(saved["segment"][0] - start) <= 0.02
                and abs(saved["segment"][1] - end) <= 0.02
                for saved in annotations
            )
            if not duplicate:
                annotations.append(annotation)
        if not annotations:
            raise ValueError(f"{video}: no valid training annotations")

        fps = float(metadata.get("fps", self.config["default_fps"]))
        offset = (
            (self.config["base_frame"] - self.config["stride"]) * 0.5 / fps
        )
        segments = torch.tensor(
            [annotation["segment"] for annotation in annotations],
            dtype=torch.float32,
        )
        segments -= offset
        labels = torch.tensor(
            [self.class_to_index[annotation["label"]] for annotation in annotations],
            dtype=torch.long,
        )
        crop_config = dict(self.config)
        crop_config["default_fps"] = fps
        if self.config["detector_training"].get("sampling") == "historical_workers":
            # DataLoader seeds each worker's Python RNG once per epoch, as in
            # the archived detector run. Do not reseed for individual videos.
            rng = random
        else:
            rng = random.Random(
                self.config["seed"] + self.epoch * 1_000_003 + index
            )
        features, segments, labels = _crop_training_features(
            features, segments, labels, crop_config, rng
        )
        info = {
            "video_name": video,
            "segments": segments,
            "labels": labels,
            "video_duration": torch.tensor(float(metadata["duration"])),
            "feature_duration": torch.tensor(
                features.shape[1] * self.config["stride"] / fps
            ),
            "fps": torch.tensor(fps),
            "base_frames": torch.tensor(self.config["base_frame"]),
            "offset": torch.tensor(offset),
            "stride": torch.tensor(self.config["stride"]),
        }
        return features, info
