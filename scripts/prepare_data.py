#!/usr/bin/env python3
"""Download and validate the pre-extracted THUMOS14 features."""

import argparse
import os
import sys
from pathlib import Path

if __name__ == "__main__" and os.environ.get("HYRCAP_MANAGED_CONSOLE") != "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from util.console import run_cli

    raise SystemExit(run_cli(__file__))

import numpy as np
from huggingface_hub import snapshot_download


SPLITS = (("val", 200), ("test", 212))
FEATURES = (
    ("InternVideo2", "feature_internvideo2_6b", 3200),
    ("V-JEPA2", "feature_vjepa", 8448),
)


def complete(root):
    return (root / "annotations.json").is_file() and all(
        len(list((root / "features" / folder / split).glob("*.npy"))) == expected
        for _, folder, _ in FEATURES
        for split, expected in SPLITS
    )


def validate_feature_set(name, root, dimension):
    for split, expected in SPLITS:
        files = sorted((root / split).glob("*.npy"))
        if len(files) != expected:
            raise RuntimeError(
                f"{name} {split}: expected {expected} files, found {len(files)}"
            )
        for path in files:
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != dimension:
                raise ValueError(
                    f"{path}: expected [T, {dimension}], got {value.shape}"
                )
        print(f"{name} {split}: {len(files)} files verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--repo", required=True)
    arguments = parser.parse_args()

    data_root = arguments.data_root.resolve()
    if not complete(data_root):
        print(f"Downloading dataset: {arguments.repo}", flush=True)
        snapshot_download(
            repo_id=arguments.repo,
            repo_type="dataset",
            local_dir=data_root,
            allow_patterns=[
                "annotations.json",
                *[
                    f"features/{folder}/{split}/*.npy"
                    for _, folder, _ in FEATURES
                    for split, _ in SPLITS
                ],
            ],
        )
    for name, folder, dimension in FEATURES:
        validate_feature_set(name, data_root / "features" / folder, dimension)
    if not (data_root / "annotations.json").is_file():
        raise FileNotFoundError(f"{arguments.repo} must contain annotations.json")
    print(f"Dataset ready: {data_root}", flush=True)


if __name__ == "__main__":
    main()
