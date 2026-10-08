#!/usr/bin/env python3
"""Train the HyrCap detector and calibration modules, or evaluate a checkpoint."""

import argparse
import hashlib
import json
import os
import platform
import random
import runpy
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__" and os.environ.get("HYRCAP_MANAGED_CONSOLE") != "1":
    from util.console import run_cli

    raise SystemExit(run_cli(__file__))

# Set library defaults before importing NumPy or PyTorch. Explicit environment
# overrides remain available, but can change numerical reproducibility.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
RUNTIME = Path(os.environ.get("HYRCAP_RUNTIME_DIR", Path.home() / ".cache/hyrcap"))
sys.path.insert(0, str(RUNTIME / "native"))
sys.path.insert(0, str(ROOT))

from datasets.action_eval import TemporalDetectionEvaluator  # noqa: E402
from datasets.thumos14 import THUMOS14TrainingDataset, load_features  # noqa: E402


IGNORED_VIDEOS = {
    "video_test_0000270",
    "video_test_0001292",
    "video_test_0001496",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    def convert(item):
        if isinstance(item, (np.ndarray, torch.Tensor)):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, default=convert)


def load_observation(video, metadata, intern_features, vjepa_features, config, device):
    feature = load_features(video, intern_features, vjepa_features)
    fps = float(metadata.get("fps", config["default_fps"]))
    offset = (config["base_frame"] - config["stride"]) * 0.5 / fps
    info = {
        "video_name": video,
        "video_duration": torch.tensor(float(metadata["duration"]), device=device),
        "feature_duration": torch.tensor(
            feature.shape[1] * config["stride"] / fps, device=device
        ),
        "fps": torch.tensor(fps, device=device),
        "base_frames": torch.tensor(config["base_frame"], device=device),
        "offset": torch.tensor(offset, device=device),
        "stride": torch.tensor(config["stride"], device=device),
        "segments": torch.empty((0, 2), device=device),
        "labels": torch.empty((0,), dtype=torch.long, device=device),
    }
    return feature, info


def collect_predictions(arguments, annotations, checkpoint, split):
    from models import build_model
    from util.misc import nested_tensor_from_tensor_list

    models = []
    for state_key, config_key in (
        ("detector", "detector_config"),
        ("reference_detector", "reference_config"),
    ):
        model, postprocess = build_model(argparse.Namespace(**checkpoint[config_key]))
        model.load_state_dict(checkpoint[state_key], strict=True)
        models.append(model.requires_grad_(False).to(arguments.device).eval())

    database = annotations["database"]
    videos = sorted(
        video
        for video, item in database.items()
        if item["subset"] == split and video not in IGNORED_VIDEOS
    )
    expected = 200 if split == "val" else 210
    if len(videos) != expected:
        raise ValueError(f"Expected {expected} THUMOS14 {split} videos, got {len(videos)}")
    for folder in (arguments.intern_features, arguments.vjepa_features):
        missing = [video for video in videos if not (folder / f"{video}.npy").is_file()]
        if missing:
            raise FileNotFoundError(
                f"{folder}: {len(missing)} missing videos; first is {missing[0]}"
            )

    records = []
    started = time.time()
    with torch.inference_mode():
        for index, video in enumerate(videos, 1):
            feature, info = load_observation(
                video,
                database[video],
                arguments.intern_features,
                arguments.vjepa_features,
                checkpoint["detector_config"],
                arguments.device,
            )
            timing = [
                info[key].reshape(1)
                for key in ("video_duration", "feature_duration", "stride", "offset")
            ]
            record = {"video_name": video, "duration": float(info["video_duration"])}
            for branch, model in zip(("primary", "auxiliary"), models):
                prediction = postprocess(
                    model(
                        nested_tensor_from_tensor_list([feature]).to(arguments.device),
                        [info],
                    ),
                    *timing,
                    checkpoint["detector_config"]["duration_thresh"],
                )[0]
                record.update(
                    {
                        f"{branch}_{key}": prediction[key]
                        for key in ("segments", "scores", "labels")
                    }
                )
                if branch == "primary":
                    record.update(
                        {key: prediction[key] for key in ("query_features", "logits")}
                    )
            records.append(record)
            if index == 1 or index % 10 == 0 or index == len(videos):
                print(
                    f"Inference {index}/{len(videos)}; {time.time() - started:.1f}s",
                    flush=True,
                )
    del models, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [
        {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in record.items()
        }
        for record in records
    ]


def evaluate(arguments, config, annotations, output, checkpoint):
    from models.hyrcap.calibration import CalibrationPipeline

    started = time.time()
    records = collect_predictions(arguments, annotations, checkpoint, "test")
    calibrator = CalibrationPipeline.from_checkpoint_data(checkpoint, arguments.device)
    predictions = calibrator.calibrate_records(records, checkpoint["calibration_scoring"])
    settings = checkpoint["evaluation_config"]
    evaluator = TemporalDetectionEvaluator(
        annotations=annotations,
        ignored_videos=IGNORED_VIDEOS,
        thresholds=settings["iou_range"],
        nms_threshold=settings["nms_thr"],
        minimum_score=settings["min_score"],
        top_k=settings["eval_topk"],
        classes=checkpoint["classes"],
        voting_iou=settings["voting_iou"],
        voting_score_offset=settings["voting_score_offset"],
    )
    videos = []
    for record, prediction in zip(records, predictions, strict=True):
        video = record["video_name"]
        videos.append(video)
        record["detector_segments"] = record["primary_segments"]
        record["detector_scores"] = record["primary_scores"]
        record.update({f"primary_{key}": value for key, value in prediction.items()})
        evaluator.update(video, prediction)
    inference_seconds = time.time() - started
    torch.save(records, output / "proposals.pt")
    evaluator.summarize()

    for mode, table in evaluator.predictions.items():
        results = {video: [] for video in videos}
        for video, start, end, score, label in table.itertuples(index=False, name=None):
            results[video].append(
                {
                    "segment": [start, end],
                    "score": score,
                    "label": checkpoint["classes"][int(label)],
                }
            )
        write_json(output / f"detections_{mode}.json", {"results": results})

    summary = {
        "mode": "full_calibrated",
        "fresh_inference": True,
        "checkpoint_format": checkpoint["format"],
        "class_agnostic_eval": False,
        "completed_videos": len(videos),
        "expected_videos": 210,
        "inference_seconds": inference_seconds,
        "metrics": evaluator.stats,
        "calibration_scoring": checkpoint["calibration_scoring"],
        "evaluation_config": settings,
        "evaluation_seconds": time.time() - started - inference_seconds,
    }
    write_json(output / "summary.json", summary)
    print(f"Detection mAP: {100 * evaluator.stats['nms']['mAP']:.4f}%", flush=True)


def training_collate(batch):
    from util.misc import nested_tensor_from_tensor_list

    return (
        nested_tensor_from_tensor_list([item[0] for item in batch]),
        [item[1] for item in batch],
    )


def train_detector(arguments, config, annotations, output):
    """Train the temporal proposal detector from random initialization."""
    from models import build_model
    from models.hyrcap.model import DetectionCriterion

    settings = dict(config["detector_training"])
    dataset = THUMOS14TrainingDataset(
        annotations["database"],
        arguments.intern_features,
        arguments.vjepa_features,
        config,
    )
    model, _ = build_model(argparse.Namespace(**config))
    criterion = DetectionCriterion(config).to(arguments.device)
    model.to(arguments.device).train()
    ema = None
    if settings.get("use_ema", False):
        ema = torch.optim.swa_utils.AveragedModel(
            model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                settings["ema_decay"]
            ),
        )
        ema.requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["lr"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, settings["lr_drop"], gamma=settings["lr_gamma"]
    )
    historical_sampling = settings.get("sampling") == "historical_workers"
    if historical_sampling:
        sampler = torch.utils.data.RandomSampler(dataset)
        batch_sampler = torch.utils.data.BatchSampler(
            sampler, settings["batch_size"], drop_last=True
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=settings["num_workers"],
            persistent_workers=False,
            pin_memory=True,
            collate_fn=training_collate,
        )

    started = time.time()
    global_step = 0
    last_loss = None
    with (output / "detector_training.jsonl").open("x") as log:
        for epoch in range(settings["epochs"]):
            dataset.epoch = epoch
            if not historical_sampling:
                generator = torch.Generator().manual_seed(config["seed"] + epoch)
                loader = DataLoader(
                    dataset,
                    batch_size=settings["batch_size"],
                    shuffle=True,
                    num_workers=settings["num_workers"],
                    generator=generator,
                    collate_fn=training_collate,
                )
            for batch_index, (samples, targets) in enumerate(loader, 1):
                samples = samples.to(arguments.device)
                targets = [
                    {
                        key: value.to(arguments.device)
                        if torch.is_tensor(value)
                        else value
                        for key, value in target.items()
                    }
                    for target in targets
                ]
                optimizer.zero_grad(set_to_none=True)
                losses = criterion(model(samples, targets), targets)
                loss = criterion.weighted_sum(losses)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite detector loss at step {global_step}"
                    )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    settings["clip_max_norm"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
                if ema is not None:
                    ema.update_parameters(model)
                global_step += 1
                last_loss = float(loss.detach())
                row = {
                    "epoch": epoch + 1,
                    "batch": batch_index,
                    "global_step": global_step,
                    "loss": last_loss,
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "videos": [target["video_name"] for target in targets],
                    "losses": {
                        key: float(value.detach()) for key, value in losses.items()
                    },
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                if batch_index == 1 or batch_index % 10 == 0:
                    print(
                        f"Detector epoch {epoch + 1}/{settings['epochs']} "
                        f"batch {batch_index}/{len(loader)} loss {last_loss:.5f}",
                        flush=True,
                    )
            scheduler.step()
            if historical_sampling and (epoch + 1) % 10 == 0:
                # The historical regular/EMA validation DataLoaders each
                # consumed a CPU base seed every ten epochs. Replay only those
                # two RNG draws; do not run test inference or inspect test GT.
                for _ in range(settings.get("historical_eval_seed_draws", 0)):
                    torch.empty((), dtype=torch.int64).random_().item()

            interval = settings.get("recovery_interval", 0)
            if interval and (epoch + 1) % interval == 0:
                recovery = output / "recovery"
                recovery.mkdir(exist_ok=True)
                with (recovery / f"epoch-{epoch + 1:03d}.pth").open("xb") as stream:
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "optimizer_steps": global_step,
                            "model": model.state_dict(),
                            "ema": ema.state_dict() if ema is not None else None,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all(),
                            "numpy_rng": np.random.get_state(),
                            "python_rng": random.getstate(),
                            "config": config,
                        },
                        stream,
                    )
                print(f"Recovery checkpoint saved at epoch {epoch + 1}", flush=True)

    selected_model = ema.module if ema is not None else model
    selection = "final_epoch_ema" if ema is not None else "final_epoch_regular"
    detector_state = {
        key: value.detach().cpu()
        for key, value in selected_model.state_dict().items()
    }
    detector_path = output / "detector.pth"
    with detector_path.open("xb") as stream:
        torch.save(
            {
                "format": "hyrcap-detector-v1",
                "detector": detector_state,
                "config": dict(config),
                "training": settings,
                "selection": selection,
            },
            stream,
        )
    if ema is not None:
        with (output / "detector_regular.pth").open("xb") as stream:
            torch.save(
                {
                    "format": "hyrcap-detector-v1",
                    "detector": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "training": settings,
                    "selection": "final_epoch_regular",
                },
                stream,
            )
    write_json(
        output / "detector_summary.json",
        {
            "training_videos": len(dataset),
            "epochs": settings["epochs"],
            "optimizer_steps": global_step,
            "selection": selection,
            "ema_updates": int(ema.n_averaged) if ema is not None else 0,
            "ema_decay": settings.get("ema_decay") if ema is not None else None,
            "last_loss": last_loss,
            "elapsed_seconds": time.time() - started,
            "model": str(detector_path),
        },
    )
    del selected_model, ema, model, criterion, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return detector_state


def train_calibration(arguments, config, annotations, output, checkpoint):
    from models.hyrcap.calibration import (
        FEATURE_NAMES,
        FEATURE_VERSION,
        fit_calibration,
    )
    from models.hyrcap.checkpoint import save_full_checkpoint

    started = time.time()
    database = annotations["database"]
    videos = sorted(video for video, item in database.items() if item["subset"] == "val")
    classes = sorted(
        {
            item["label"]
            for video in videos
            for item in database[video]["annotations"]
            if item["label"] != "Ambiguous"
        }
    )
    if len(classes) != config["num_classes"]:
        raise ValueError(f"Invalid classes: {len(classes)}")
    records = collect_predictions(arguments, annotations, checkpoint, "val")
    torch.save(records, output / "training_predictions.pt")
    training_config = dict(config["calibration_training"])
    calibrator, history = fit_calibration(
        records,
        database,
        training_config,
        config["calibration_scoring"],
        classes,
        arguments.device,
    )
    write_json(output / "training_history.json", history)
    evaluation_config = {
        key: config[key]
        for key in (
            "iou_range", "nms_thr", "min_score", "eval_topk",
            "voting_iou", "voting_score_offset",
        )
    }
    checkpoint_path = save_full_checkpoint(
        output / "hyrcap.pth",
        checkpoint,
        calibrator,
        training_config,
        config["calibration_scoring"],
        evaluation_config,
        classes,
        metadata={
            "training_scope": "CPCC, LQE, PRS; detector frozen",
            "training_videos": videos,
            "feature_version": FEATURE_VERSION,
            "feature_names": FEATURE_NAMES,
        },
    )
    write_json(
        output / "summary.json",
        {
            "mode": "train",
            "training_scope": "CPCC, LQE, PRS; detector frozen",
            "training_videos": len(videos),
            "epochs": training_config["epochs"],
            "checkpoint": str(checkpoint_path),
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"Unified checkpoint: {checkpoint_path}", flush=True)


def train(arguments, config, annotations, output):
    frozen = bool(arguments.detector_checkpoint)
    total_epochs = config["calibration_training"]["epochs"]
    if not frozen:
        total_epochs += 2 * config["detector_training"]["epochs"]
    print(f"HYRCAP_TRAINING_EPOCHS={total_epochs}", flush=True)
    checkpoint = {}
    for state_key, config_key, backbone, path in (
        ("detector", "detector_config", "single", arguments.detector_checkpoint),
        (
            "reference_detector", "reference_config", "fusion",
            arguments.reference_checkpoint,
        ),
    ):
        detector_config = dict(config)
        if path:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            detector_config.update(saved.get("config", saved.get("detector_config", {})))
            state = saved["detector"]
        detector_config["backbone"] = backbone
        detector_config["feature_dim"] = (
            config["modality_dims"][0]
            if backbone == "single" else sum(config["modality_dims"])
        )
        if not path:
            folder = output / state_key
            folder.mkdir()
            torch.manual_seed(config["seed"])
            np.random.seed(config["seed"])
            random.seed(config["seed"])
            state = train_detector(arguments, detector_config, annotations, folder)
        checkpoint[state_key] = state
        checkpoint[config_key] = detector_config
    train_calibration(arguments, config, annotations, output, checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), required=True)
    parser.add_argument("--config", default=str(ROOT / "config/hyrcap/thumos14.py"))
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--detector-checkpoint", default=os.environ.get("HYRCAP_DETECTOR_CHECKPOINT")
    )
    parser.add_argument(
        "--reference-checkpoint", default=os.environ.get("HYRCAP_REFERENCE_CHECKPOINT")
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default=os.environ.get("HYRCAP_DATA_ROOT"))
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.mode == "train" and (
        bool(arguments.detector_checkpoint) != bool(arguments.reference_checkpoint)
    ):
        parser.error("--detector-checkpoint and --reference-checkpoint are required together")

    config = dict(runpy.run_path(arguments.config)["config"])
    data_root = Path(arguments.data_root or config["data_root"]).resolve()
    split = "val" if arguments.mode == "train" else "test"
    arguments.intern_features = data_root / config["intern_features"] / split
    arguments.vjepa_features = data_root / config["vjepa_features"] / split
    annotation_path = data_root / config["annotations"]
    output = Path(arguments.output_dir).resolve()
    if output.is_relative_to(ROOT):
        parser.error("--output-dir must be outside the source repository")
    output.mkdir(parents=True, exist_ok=False)

    checkpoint = None
    if arguments.mode == "eval":
        if not arguments.checkpoint:
            parser.error("--checkpoint is required for --mode eval")
        checkpoint = torch.load(
            arguments.checkpoint, map_location="cpu", weights_only=True
        )
        from models.hyrcap.checkpoint import validate_full_checkpoint

        validate_full_checkpoint(checkpoint)
    elif arguments.checkpoint:
        parser.error("--checkpoint is not used by from-scratch training")
    annotations = json.loads(annotation_path.read_text())

    torch.set_num_threads(int(config.get("cpu_threads", 8)))
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "runtime_threads": {
            "torch": torch.get_num_threads(),
            **{
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
        },
        "arguments": {
            **vars(arguments),
            "intern_features": str(arguments.intern_features),
            "vjepa_features": str(arguments.vjepa_features),
        },
        "config": config,
        "annotation_sha256": sha256(annotation_path),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for folder in ("models", "datasets", "util", "config")
            for path in sorted((ROOT / folder).rglob("*.py"))
        },
    }
    if arguments.checkpoint:
        manifest["checkpoint_sha256"] = sha256(arguments.checkpoint)
    if arguments.mode == "train" and arguments.detector_checkpoint:
        manifest["detector_sha256"] = sha256(arguments.detector_checkpoint)
        manifest["reference_sha256"] = sha256(arguments.reference_checkpoint)
    manifest["source_sha256"]["main.py"] = sha256(ROOT / "main.py")
    write_json(output / "manifest.json", manifest)
    if arguments.mode == "train":
        train(arguments, config, annotations, output)
    else:
        evaluate(arguments, config, annotations, output, checkpoint)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
