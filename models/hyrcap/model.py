"""HyrCap detector architecture required by the released checkpoint."""

import copy
import math

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.misc import NestedTensor, nested_tensor_from_tensor_list
from util.segment_ops import segment_cw_to_t1t2

from .position_encoding import build_position_encoding
from .transformer import build_deformable_transformer
from .two_stream_modules import DualStreamProjConcatBackbone, SingleStreamBackbone
from .utils import MLP, get_feature_grids


class HyrCap(nn.Module):
    """Two-stream temporal proposal detector."""

    def __init__(self, position_embedding, transformer, config):
        super().__init__()
        hidden_dim = transformer.d_model
        self.transformer = transformer
        self.position_embedding = position_embedding
        self.transformer.position_embedding = position_embedding
        if getattr(config, "backbone", "fusion") == "single":
            self.input_proj = SingleStreamBackbone(
                config.modality_dims[0], hidden_dim, config.emb_dropout
            )
        else:
            self.input_proj = DualStreamProjConcatBackbone(
                hidden_dim=hidden_dim,
                modality_dims=config.modality_dims,
                dropout=config.emb_dropout,
                use_concat_residual=config.use_concat_residual,
            )

        prior_probability = 0.01
        bias = -math.log((1 - prior_probability) / prior_probability)
        class_head = nn.Linear(hidden_dim, config.num_classes)
        nn.init.constant_(class_head.bias, bias)
        encoder_class_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(encoder_class_head.bias, bias)
        self.class_embed = nn.ModuleList(
            [class_head for _ in range(transformer.decoder.num_layers)]
        )
        self.class_embed.append(encoder_class_head)
        self.transformer.decoder.class_embed = self.class_embed

        segment_head = MLP(hidden_dim, hidden_dim, 2, num_layers=3)
        nn.init.zeros_(segment_head.layers[-1].weight)
        nn.init.zeros_(segment_head.layers[-1].bias)
        self.segment_embed = nn.ModuleList(
            [segment_head for _ in range(transformer.decoder.num_layers)]
        )
        self.segment_embed.append(copy.deepcopy(segment_head))
        self.transformer.decoder.segment_embed = self.segment_embed

    def forward(self, samples, info):
        if not isinstance(samples, NestedTensor):
            samples = (
                NestedTensor(*samples)
                if isinstance(samples, (list, tuple))
                else nested_tensor_from_tensor_list(samples)
            )

        sources, masks = self.input_proj(samples.tensors, samples.mask)
        fps = torch.stack([item["fps"] for item in info])
        stride = torch.stack([item["stride"] for item in info])
        feature_durations = torch.stack([item["feature_duration"] for item in info])
        sources = [sources[0] + self.position_embedding(fps.unsqueeze(-1))]
        grids = [get_feature_grids(masks[0], fps, stride, stride)]
        positions = [self.position_embedding(grids[0])]

        (
            hidden_states,
            intermediate_segments,
            encoder_logits,
            encoder_segments,
            encoder_mask,
            _,
            query_mask,
        ) = self.transformer(
            sources,
            masks,
            positions,
            grids,
            feature_durations,
            fps,
            stride,
            None,
        )
        logits = torch.stack(
            [
                self.class_embed[level](hidden_states[level])
                for level in range(hidden_states.shape[0])
            ]
        )
        output = {
            "pred_logits": logits[-1],
            "pred_segments": intermediate_segments[-1],
            "query_features": hidden_states[-1],
            "mask": query_mask,
        }
        output["aux_outputs"] = [
            {
                "pred_logits": level_logits,
                "pred_segments": level_segments,
                "mask": query_mask,
            }
            for level_logits, level_segments in zip(
                logits[:-1], intermediate_segments[:-1], strict=True
            )
        ]
        output["enc_outputs"] = {
            "pred_logits": encoder_logits,
            "pred_segments": encoder_segments,
            "mask": encoder_mask,
        }
        return output


class PostProcess(nn.Module):
    """Convert detector tensors to scored temporal segments in seconds."""

    @torch.no_grad()
    def forward(
        self,
        outputs,
        video_durations,
        feature_durations,
        strides,
        offsets,
        duration_threshold=0.05,
    ):
        del feature_durations, strides
        center_width = torch.cat(
            (
                outputs["pred_segments"][..., :1],
                outputs["pred_segments"][..., 1:].exp(),
            ),
            dim=-1,
        )
        segments = segment_cw_to_t1t2(center_width) + offsets[:, None, None]
        results = []
        for index in range(len(segments)):
            current_segments = segments[index].clip(0, video_durations[index].item())
            valid = (
                current_segments[:, 1] - current_segments[:, 0]
            ) > duration_threshold
            valid &= ~outputs["mask"][index]
            logits = outputs["pred_logits"][index][valid].cpu()
            scores, labels = logits.sigmoid().max(dim=-1)
            results.append(
                {
                    "scores": scores,
                    "labels": labels,
                    "segments": current_segments[valid].cpu(),
                    "logits": logits,
                    "query_features": outputs["query_features"][index][valid].cpu(),
                }
            )
        return results


def _segments_to_start_end(segments):
    center = segments[..., 0]
    width = segments[..., 1].exp()
    return torch.stack((center - 0.5 * width, center + 0.5 * width), dim=-1)


def _pairwise_iou(first, second):
    left = torch.maximum(first[:, None, 0], second[None, :, 0])
    right = torch.minimum(first[:, None, 1], second[None, :, 1])
    intersection = (right - left).clamp_min(0)
    first_length = (first[:, 1] - first[:, 0]).clamp_min(0)
    second_length = (second[:, 1] - second[:, 0]).clamp_min(0)
    union = first_length[:, None] + second_length[None, :] - intersection
    return intersection / union.clamp_min(1e-8)


class DetectionCriterion(nn.Module):
    """Hungarian matching with focal classification and temporal IoU losses."""

    def __init__(self, config):
        super().__init__()
        self.class_cost = float(config["set_cost_class"])
        self.iou_cost = float(config["set_cost_iou"])
        self.class_weight = float(config["cls_loss_coef"])
        self.iou_weight = float(config["iou_loss_coef"])
        self.alpha = float(config["focal_alpha"])
        self.gamma = 2.0

    @torch.no_grad()
    def _match(self, outputs, targets, encoder=False):
        matches = []
        for batch_index, target in enumerate(targets):
            valid = ~outputs["mask"][batch_index]
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
            probabilities = outputs["pred_logits"][batch_index, valid].sigmoid()
            labels = target["labels"]
            negative = (1 - self.alpha) * probabilities.pow(self.gamma) * (
                -(1 - probabilities + 1e-8).log()
            )
            positive = self.alpha * (1 - probabilities).pow(self.gamma) * (
                -(probabilities + 1e-8).log()
            )
            class_cost = positive[:, labels] - negative[:, labels]
            predicted = _segments_to_start_end(
                outputs["pred_segments"][batch_index, valid]
            )
            iou_cost = -_pairwise_iou(predicted, target["segments"])
            class_weight = 1.0 if encoder else self.class_cost
            cost = class_weight * class_cost + self.iou_cost * iou_cost
            source, destination = linear_sum_assignment(
                cost.detach().cpu().numpy()
            )
            source = torch.as_tensor(source, dtype=torch.long, device=cost.device)
            destination = torch.as_tensor(
                destination, dtype=torch.long, device=cost.device
            )
            matches.append((valid_indices[source], destination))
        return matches

    def _losses(self, outputs, targets, matches, normalizer):
        logits = outputs["pred_logits"]
        target_logits = torch.zeros_like(logits)
        for batch_index, (source, destination) in enumerate(matches):
            target_logits[
                batch_index, source, targets[batch_index]["labels"][destination]
            ] = 1

        probability = logits.sigmoid()
        cross_entropy = F.binary_cross_entropy_with_logits(
            logits, target_logits, reduction="none"
        )
        probability_target = (
            probability * target_logits + (1 - probability) * (1 - target_logits)
        )
        focal = cross_entropy * (1 - probability_target).pow(self.gamma)
        alpha = self.alpha * target_logits + (1 - self.alpha) * (1 - target_logits)
        focal *= alpha
        valid = (~outputs["mask"]).unsqueeze(-1)
        focal *= valid.to(focal.dtype)
        query_count = valid.sum(1).clamp_min(1).to(focal.dtype)
        loss_class = (
            (focal.sum(1) / query_count).sum() / normalizer * query_count.mean()
        )

        predicted = []
        expected = []
        for batch_index, (source, destination) in enumerate(matches):
            predicted.append(outputs["pred_segments"][batch_index, source])
            expected.append(targets[batch_index]["segments"][destination])
        predicted = _segments_to_start_end(torch.cat(predicted))
        expected = torch.cat(expected)
        loss_iou = (1 - _pairwise_iou(predicted, expected).diag()).sum() / normalizer
        return loss_class, loss_iou

    def forward(self, outputs, targets):
        normalizer = max(sum(len(target["labels"]) for target in targets), 1)
        losses = {}

        matches = self._match(outputs, targets)
        loss_class, loss_iou = self._losses(
            outputs, targets, matches, normalizer
        )
        losses["loss_ce"] = loss_class
        losses["loss_iou"] = loss_iou

        for index, auxiliary in enumerate(outputs.get("aux_outputs", [])):
            matches = self._match(auxiliary, targets)
            loss_class, loss_iou = self._losses(
                auxiliary, targets, matches, normalizer
            )
            losses[f"loss_ce_{index}"] = loss_class
            losses[f"loss_iou_{index}"] = loss_iou

        encoder = outputs.get("enc_outputs")
        if encoder is not None:
            binary_targets = [
                {
                    "segments": target["segments"],
                    "labels": torch.zeros_like(target["labels"]),
                }
                for target in targets
            ]
            matches = self._match(encoder, binary_targets, encoder=True)
            loss_class, loss_iou = self._losses(
                encoder, binary_targets, matches, normalizer
            )
            losses["loss_ce_enc"] = loss_class
            losses["loss_iou_enc"] = loss_iou
        return losses

    def weighted_sum(self, losses):
        return sum(
            self.class_weight * value
            if key.startswith("loss_ce")
            else self.iou_weight * value
            for key, value in losses.items()
        )


def build(config):
    position_embedding = build_position_encoding(config)
    transformer = build_deformable_transformer(config)
    return HyrCap(position_embedding, transformer, config), PostProcess()
