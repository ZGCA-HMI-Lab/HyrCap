#!/usr/bin/env python3
"""Run the complete HyrCap training and evaluation pipeline."""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__" and os.environ.get("HYRCAP_MANAGED_CONSOLE") != "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from util.console import run_cli

    raise SystemExit(run_cli(__file__))


def run(command, root):
    subprocess.run(command, cwd=root, env=os.environ.copy(), check=True)


def ensure_inputs(data_root):
    required = (
        data_root / "annotations.json",
        data_root / "features" / "feature_internvideo2_6b" / "val",
        data_root / "features" / "feature_internvideo2_6b" / "test",
        data_root / "features" / "feature_vjepa" / "val",
        data_root / "features" / "feature_vjepa" / "test",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset input: {missing[0]}. "
            "Run bash scripts/data_download.sh first."
        )


def ensure_operators(root):
    probe = (
        "from models.hyrcap.ops.functions.ms_deform_attn_func import MSDA; "
        "from util import nms"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=root,
        env=os.environ.copy(),
        check=False,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        run(["bash", str(root / "scripts" / "build_ops.sh")], root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    data_root = Path(os.environ["HYRCAP_DATA_ROOT"]).resolve()
    ensure_inputs(data_root)
    ensure_operators(root)

    output_root = Path(
        os.environ.get("HYRCAP_OUTPUT_ROOT", str(root.parent / "hyrcap-run-outputs"))
    ).expanduser().resolve()
    if output_root.is_relative_to(root):
        parser.error("HYRCAP_OUTPUT_ROOT must be outside the source repository")
    run_root = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    train_output = run_root / "train"
    eval_output = run_root / "eval"
    common = [
        sys.executable,
        "-B",
        str(root / "main.py"),
        "--config",
        str(root / "config" / "hyrcap" / "thumos14.py"),
        "--data-root",
        str(data_root),
    ]
    run(common + ["--mode", "train", "--output-dir", str(train_output)], root)
    run(
        common
        + [
            "--mode",
            "eval",
            "--checkpoint",
            str(train_output / "hyrcap.pth"),
            "--output-dir",
            str(eval_output),
        ],
        root,
    )
    print(f"Training output: {train_output}")
    print(f"Evaluation output: {eval_output}")


if __name__ == "__main__":
    main()
