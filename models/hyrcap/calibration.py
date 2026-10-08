"""Proposal calibration used by HyrCap training and evaluation."""

from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .modules import CPCC, LQE, PRS


FEATURE_VERSION = "hyrcap-prediction-calibration-v5"
SCALAR_FEATURE_NAMES = [
    "score",
    "logit_score",
    "log_duration",
    "relative_duration",
    "relative_center",
    "auxiliary_support",
    "auxiliary_best_score",
    "auxiliary_top3_iou",
    "auxiliary_score_weighted_support",
    "stronger_neighbor_iou",
    "overlap_density",
    "relative_score_rank",
    "left_boundary_support",
    "right_boundary_support",
    "same_class_auxiliary_support",
    "video_log_duration",
]
FEATURE_NAMES = (
    SCALAR_FEATURE_NAMES
    + [f"query_{index}" for index in range(512)]
    + [f"logit_{index}" for index in range(20)]
)


def temporal_iou(first, second):
    first = np.asarray(first, np.float64)
    second = np.asarray(second, np.float64)
    intersection = np.maximum(
        0,
        np.minimum(first[:, None, 1], second[None, :, 1])
        - np.maximum(first[:, None, 0], second[None, :, 0]),
    )
    union = (
        (first[:, 1] - first[:, 0])[:, None]
        + (second[:, 1] - second[:, 0])[None, :]
        - intersection
    )
    return intersection / np.maximum(union, 1e-8)


def prepare_prediction(record, scoring):
    """Construct checkpoint features from detector predictions without GT."""
    segments = np.asarray(record["primary_segments"], np.float32)
    scores = np.asarray(record["primary_scores"], np.float32)
    labels = np.asarray(record["primary_labels"], np.int64)
    auxiliary = np.asarray(record["auxiliary_segments"], np.float32)
    auxiliary_scores = np.asarray(record["auxiliary_scores"], np.float32)
    auxiliary_labels = np.asarray(record["auxiliary_labels"], np.int64)
    if len(segments) and len(auxiliary):
        overlap = temporal_iou(segments, auxiliary)
        weights = np.where(
            (labels[:, None] == auxiliary_labels[None, :])
            & (overlap >= scoring["boundary_iou"])
            & (auxiliary_scores[None, :] > scoring["boundary_min_score"]),
            overlap * auxiliary_scores[None, :] ** scoring["support_power"],
            0.0,
        )
        count = min(scoring["boundary_topk"], weights.shape[1])
        indices = np.argsort(-weights, axis=1, kind="stable")[:, :count]
        retained = np.take_along_axis(weights, indices, axis=1)
        mass = retained.sum(1)
        voted = (auxiliary[indices] * retained[:, :, None]).sum(1) / np.maximum(
            mass[:, None], 1e-8
        )
        best_score = np.where(retained > 0, auxiliary_scores[indices], 0.0).max(1)
        fraction = scoring["boundary_alpha"] * best_score / np.maximum(
            best_score + scores, 1e-6
        )
        fraction[mass == 0] = 0.0
        segments = (
            segments * (1 - fraction[:, None]) + voted * fraction[:, None]
        ).astype(np.float32)
    if not len(auxiliary):
        auxiliary = np.zeros((1, 2), np.float32)
        auxiliary_scores = np.zeros(1, np.float32)
        auxiliary_labels = np.full(1, -1, np.int64)
    duration = float(record["duration"])
    length = np.maximum(segments[:, 1] - segments[:, 0], 1e-5)
    center = segments.mean(1)

    overlap = temporal_iou(segments, auxiliary).astype(np.float32)
    support = CPCC.compute_support(segments, auxiliary)
    best = overlap.argmax(1)
    best_score = auxiliary_scores[best]
    top3 = np.sort(overlap, axis=1)[:, -min(3, overlap.shape[1]) :].mean(1)
    weighted = (overlap * auxiliary_scores[None, :]).max(1)

    same = temporal_iou(segments, segments).astype(np.float32)
    np.fill_diagonal(same, 0)
    stronger = (scores[None, :] > scores[:, None]) & (
        labels[None, :] == labels[:, None]
    )
    stronger_support = (
        np.where(stronger, same, 0).max(1) if len(scores) else np.empty(0, np.float32)
    )
    density = (same >= 0.5).sum(1) / max(len(scores), 1)
    rank = np.argsort(np.argsort(-scores)).astype(np.float32) / max(len(scores), 1)

    same_auxiliary = np.where(
        labels[:, None] == auxiliary_labels[None, :], overlap, 0
    ).max(1)

    left_support = np.exp(-np.abs(segments[:, 0] - auxiliary[best, 0]) / length)
    right_support = np.exp(-np.abs(segments[:, 1] - auxiliary[best, 1]) / length)
    bounded = np.clip(scores, 1e-6, 1 - 1e-6)
    features = np.stack(
        [
            scores,
            np.log(bounded / (1 - bounded)),
            np.log1p(length),
            length / duration,
            center / duration,
            support,
            best_score,
            top3,
            weighted,
            stronger_support,
            density,
            rank,
            left_support,
            right_support,
            same_auxiliary,
            np.full_like(scores, np.log1p(duration)),
        ],
        axis=1,
    ).astype(np.float32)
    support = (
        temporal_iou(segments, auxiliary)
        * auxiliary_scores[None, :] ** scoring["support_power"]
    ).max(1).astype(np.float32)
    features[:, 5] = support
    features = np.concatenate(
        (features, np.asarray(record["query_features"]), np.asarray(record["logits"])),
        axis=1,
    )
    if features.shape != (len(scores), len(FEATURE_NAMES)):
        raise ValueError(f"Invalid calibration features: {features.shape}")
    table = pd.DataFrame(
        {
            "video_id": record["video_name"],
            "pred_label": labels,
            "t_start": segments[:, 0],
            "t_end": segments[:, 1],
            "detector_score": scores,
            "segment_center": center,
            "segment_len": length,
        }
    )
    edges, edge_features = PRS.build_graph(
        table, edge_iou=0.5, topk=8, higher_only=True
    )
    return {
        "video_name": record["video_name"],
        "duration": duration,
        "segments": torch.from_numpy(segments),
        "scores": torch.from_numpy(scores),
        "labels": torch.from_numpy(labels),
        "features": torch.from_numpy(features),
        "support": torch.from_numpy(support),
        "edges": torch.from_numpy(edges),
        "edge_features": torch.from_numpy(edge_features),
    }


def add_training_targets(record, annotations, classes, threshold=0.5):
    """Build localization-quality and duplicate targets from validation GT."""
    targets = np.asarray([item["segment"] for item in annotations], np.float64)
    if targets.size == 0:
        raise ValueError(f'{record["video_name"]}: validation video has no GT events')
    overlap = temporal_iou(record["segments"], targets)
    labels = np.asarray([classes.index(item["label"]) for item in annotations])
    overlap = np.where(record["labels"].numpy()[:, None] == labels[None, :], overlap, 0)
    quality = overlap.max(1).astype(np.float32)
    matched = overlap.argmax(1)
    keep = np.ones(len(quality), np.float32)
    scores = record["scores"].numpy()
    for target in range(len(targets)):
        indices = np.flatnonzero((matched == target) & (quality >= threshold))
        if len(indices):
            keep[indices] = 0
            best = indices[np.argmax(quality[indices] + 0.05 * scores[indices])]
            keep[best] = 1
    return {
        **record,
        "quality_target": torch.from_numpy(quality),
        "keep_target": torch.from_numpy(keep),
    }


def batch_records(records, device, training=False):
    keys = ["features", "scores", "support"]
    if training:
        keys += ["quality_target", "keep_target"]
    batch = {key: torch.cat([record[key] for record in records]).to(device) for key in keys}
    edges = []
    edge_features = []
    offset = 0
    for record in records:
        edges.append(record["edges"] + offset)
        edge_features.append(record["edge_features"])
        offset += len(record["scores"])
    batch["edges"] = torch.cat(edges, dim=1).to(device)
    batch["edge_features"] = torch.cat(edge_features).to(device)
    return batch


class CalibrationPipeline(nn.Module):
    """Apply the three released proposal-scoring modules."""

    def __init__(
        self, feature_mean, feature_std, hidden=64, graph_hidden=96, dropout=0.1
    ):
        super().__init__()
        self.cpcc = CPCC(learned=True, dropout=dropout)
        self.lqe = LQE(len(SCALAR_FEATURE_NAMES), hidden_dim=hidden, dropout=dropout)
        self.prs = PRS(3, hidden_dim=graph_hidden, dropout=dropout)
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean).float())
        self.register_buffer(
            "feature_std", torch.as_tensor(feature_std).float().clamp_min(0.01)
        )
        self.lqe = LQE(len(feature_mean), hidden_dim=hidden, dropout=dropout)

    @classmethod
    def from_checkpoint_data(cls, checkpoint, device):
        if checkpoint.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"Calibration checkpoint requires feature version {FEATURE_VERSION}."
            )
        if checkpoint.get("modules_enabled") != {
            "cpcc": True,
            "lqe": True,
            "prs": True,
        }:
            raise ValueError("Expected CPCC, LQE, and PRS weights.")
        config = checkpoint["calibration_config"]
        state = checkpoint["calibrator"]
        model = cls(
            state["feature_mean"],
            state["feature_std"],
            hidden=config["hidden"],
            dropout=config["dropout"],
        ).to(device)
        model.load_state_dict(state, strict=True)
        return model.eval()

    @torch.inference_mode()
    def calibrate_records(self, records, scoring):
        prepared = [prepare_prediction(record, scoring) for record in records]
        support = np.concatenate([record["support"].numpy() for record in prepared])
        active = np.zeros(len(support), dtype=bool)
        count = round(len(support) * scoring["cpcc_ratio"])
        active[np.argsort(-support, kind="stable")[:count]] = True
        predictions = []
        offset = 0
        for record in prepared:
            count = len(record["scores"])
            output = self(batch_records([record], self.feature_mean.device))
            score = self.score(
                record,
                output["quality"].cpu().numpy(),
                output["keep"].cpu().numpy(),
                output["gate"].cpu().numpy(),
                active[offset : offset + count],
                scoring,
            )
            predictions.append(
                {
                    "segments": record["segments"],
                    "scores": torch.from_numpy(score),
                    "labels": record["labels"],
                }
            )
            offset += count
        return predictions

    def forward(self, batch):
        features = ((batch["features"] - self.feature_mean) / self.feature_std).clamp(
            -10, 10
        )
        gate_logit = self.cpcc(batch["features"][:, [0, 5, 6, 3]])
        quality_logit = self.lqe(features)
        gate = gate_logit.sigmoid()
        quality = quality_logit.sigmoid()
        nodes = torch.stack([batch["scores"], features[:, 2], quality], dim=1)
        rank, winner, duplicate = self.prs(
            nodes, batch["edges"], batch["edge_features"]
        )
        return {
            "gate_logit": gate_logit,
            "gate": gate,
            "quality_logit": quality_logit,
            "quality": quality,
            "rank_logit": rank,
            "winner_logit": winner,
            "duplicate_logit": duplicate,
            "keep": (-duplicate).sigmoid(),
        }

    @staticmethod
    def loss(output, batch, class_weights):
        quality = batch["quality_target"]
        keep = batch["keep_target"]
        duplicate = 1 - keep
        balance = (1 - duplicate.mean()).clamp_min(0.01) / duplicate.mean().clamp_min(0.001)
        weights = torch.where(
            duplicate > 0.5,
            balance.clamp(max=20),
            torch.ones_like(keep),
        )
        def bce(logits, target):
            return F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        loss = (
            bce(output["quality_logit"], quality)
            + 0.25 * bce(output["gate_logit"], quality)
            + 0.1 * bce(output["rank_logit"], quality)
        )
        loss += weights * (
            0.25 * bce(output["duplicate_logit"], duplicate)
            + 0.1 * bce(output["winner_logit"], keep)
        )
        return (loss * class_weights).sum() / class_weights.sum()

    def score(self, record, quality, keep, gate, active, scoring):
        score = record["scores"].numpy().astype(np.float64).copy()
        score[active] *= (
            1 + scoring["cpcc_beta"] * record["support"].numpy()[active]
            * (0.98 + 0.02 * gate[active])
        )
        score *= np.maximum(quality, 1e-6) ** scoring["lqe_gamma"]
        score *= np.maximum(keep, 1e-6) ** scoring["prs_gamma"]
        return score


def fit_calibration(records, database, config, scoring, classes, device):
    """Train CPCC, LQE, and PRS from frozen-detector validation predictions."""
    if not records or any(
        database[record["video_name"]]["subset"] != "val" for record in records
    ):
        raise ValueError("Calibration training requires THUMOS14 val videos only.")

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    prepared = [
        add_training_targets(
            prepare_prediction(record, scoring),
            database[record["video_name"]]["annotations"],
            classes,
            threshold=config["threshold"],
        )
        for record in records
    ]
    features = torch.cat([record["features"] for record in prepared])
    model = CalibrationPipeline(
        features.mean(0),
        features.std(0),
        hidden=config["hidden"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config["epochs"], eta_min=config["lr"] * config["min_lr_ratio"]
    )
    counts = Counter(
        item["label"]
        for video in {record["video_name"] for record in records}
        for item in database[video]["annotations"]
    )
    weights = torch.tensor(
        [1 / np.sqrt(max(counts[name], 1)) for name in classes], dtype=torch.float32
    )
    weights = (weights / weights.mean()).clamp(0.25, 4)
    history = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        indices = np.random.permutation(len(prepared))
        losses = []
        for start in range(0, len(indices), config["batch_size"]):
            chosen = [
                prepared[index]
                for index in indices[start : start + config["batch_size"]]
            ]
            batch = batch_records(chosen, device, training=True)
            class_weights = torch.cat(
                [weights[record["labels"]] for record in chosen]
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(model(batch), batch, class_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite calibration loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == config["epochs"]:
            print(f"Calibration {row}", flush=True)
    return model.eval(), history
