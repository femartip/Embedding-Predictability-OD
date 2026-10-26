from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion
from torchvision.ops import box_convert, box_iou
import numpy as np

class LocalizationRecallPrecision(Metric):
    """TorchMetrics-compatible implementation of the Localization Recall Precision (LRP) error."""
    def __init__(self, tau: float = 0.5, box_format: str = "xyxy", **kwargs) -> None:
        super().__init__(**kwargs)
        assert 0.0 < tau < 1.0, f"IoU threshold `tau` must lie in (0, 1). Got {tau}"
        assert box_format in {"xyxy", "xywh", "cxcywh"}, f"Unsupported `box_format`: {box_format}"

        self.tau = tau
        self.box_format = box_format

        self.add_state("loc_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("tp_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: Sequence[dict[str, Tensor]] | dict[str, Tensor],
        target: Sequence[dict[str, Tensor]] | dict[str, Tensor],
    ) -> None:
        preds_list = self._ensure_sequence(preds)
        target_list = self._ensure_sequence(target)
        assert len(preds_list) == len(target_list), "Number of predictions and targets must match."

        for pred_dict, target_dict in zip(preds_list, target_list):
            loc_error, fp, fn, tp = self._process_single(pred_dict, target_dict)
            self.loc_error += loc_error
            self.fp_total += fp
            self.fn_total += fn
            self.tp_total += tp

    def compute(self) -> Tensor:
        total = self.tp_total + self.fp_total + self.fn_total  #type: ignore
        if total == 0:
            return torch.tensor(0.0)

        loc_term = torch.tensor(0.0)
        if self.tp_total > 0:
            loc_term = self.loc_error / (1.0 - self.tau)

        lrp_error = (loc_term + self.fp_total + self.fn_total) / total
        return torch.clamp(lrp_error, min=0.0, max=1.0)

    def _process_single(
        self,
        pred_dict: dict[str, Tensor],
        target_dict: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pred_boxes = self._prepare_boxes(pred_dict.get("boxes"))
        pred_scores = self._prepare_scores(pred_dict.get("scores"), len(pred_boxes))
        pred_labels = self._prepare_labels(pred_dict.get("labels"))
        target_boxes = self._prepare_boxes(target_dict.get("boxes"))
        target_labels = self._prepare_labels(target_dict.get("labels"))

        all_classes = torch.cat([pred_labels.unique(), target_labels.unique()], dim=0).unique()

        loc_error = torch.tensor(0.0)
        fp = torch.tensor(0.0)
        fn = torch.tensor(0.0)
        tp = torch.tensor(0.0)

        for cls in all_classes:
            cls_pred_mask = pred_labels == cls
            cls_target_mask = target_labels == cls

            cls_pred_boxes = pred_boxes[cls_pred_mask]
            cls_pred_scores = pred_scores[cls_pred_mask]
            cls_target_boxes = target_boxes[cls_target_mask]

            order = torch.argsort(cls_pred_scores, descending=True)
            cls_pred_boxes = cls_pred_boxes[order]

            ious = box_iou(cls_pred_boxes, cls_target_boxes)
            matched_targets = torch.zeros(len(cls_target_boxes), dtype=torch.bool)

            for i in range(len(cls_pred_boxes)):
                iou_vals = ious[i].masked_fill(matched_targets, -1.0)
                if iou_vals.numel() == 0:
                    fp += 1.0
                    continue
                max_iou, gt_idx = iou_vals.max(dim=0)
                if max_iou >= self.tau and not matched_targets[gt_idx]:
                    matched_targets[gt_idx] = True
                    tp += 1.0
                    loc_error += 1.0 - max_iou
                else:
                    fp += 1.0

            fn += torch.tensor(float((~matched_targets).sum().item()))

        return loc_error, fp, fn, tp

    @staticmethod
    def _ensure_sequence(data: Sequence[dict[str, Tensor]] | dict[str, Tensor] | None) -> list[dict[str, Tensor]]:
        if data is None:
            return []
        if isinstance(data, dict):
            return [data]
        return list(data)

    def _prepare_boxes(self, boxes: Tensor | Iterable[Sequence[float]] | None) -> Tensor:
        if boxes is None:
            return torch.zeros((0, 4), dtype=torch.float32)
        tensor_boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if tensor_boxes.ndim != 2 or tensor_boxes.size(-1) != 4:
            msg = f"Boxes must have shape (N, 4). Got {tensor_boxes.shape}"
            raise ValueError(msg)
        if tensor_boxes.numel() == 0:
            return tensor_boxes
        if self.box_format != "xyxy":
            tensor_boxes = box_convert(tensor_boxes, in_fmt=self.box_format, out_fmt="xyxy")
        return tensor_boxes

    @staticmethod
    def _prepare_scores(scores: Tensor | Iterable[float] | None, num_boxes: int) -> Tensor:
        if scores is None:
            return torch.ones(num_boxes, dtype=torch.float32)
        tensor_scores = torch.as_tensor(scores, dtype=torch.float32)
        if tensor_scores.ndim != 1:
            msg = f"Scores must be a 1-D tensor. Got {tensor_scores.shape}"
            raise ValueError(msg)
        if len(tensor_scores) != num_boxes:
            msg = "Number of scores must match number of boxes."
            raise ValueError(msg)
        return tensor_scores

    @staticmethod
    def _prepare_labels(labels: Tensor | Iterable[int] | None) -> Tensor:
        if labels is None:
            return torch.zeros(0, dtype=torch.long)
        tensor_labels = torch.as_tensor(labels, dtype=torch.long)
        if tensor_labels.ndim != 1:
            msg = f"Labels must be a 1-D tensor. Got {tensor_labels.shape}"
            raise ValueError(msg)
        return tensor_labels


@torch.no_grad()
def matched_iou(preds, targets):
    all_ious = []
    n_images = max(len(preds), len(targets))

    for i in range(n_images):
        p = preds[i] if i < len(preds) else {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long), "scores": torch.zeros(0)}
        t = targets[i] if i < len(targets) else {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)}
        if p["boxes"].numel() == 0 or t["boxes"].numel() == 0:
            all_ious.extend([0.0] * max(len(p["boxes"]), len(t["boxes"])))
            continue

        p_boxes = p["boxes"].float()
        p_labels = p["labels"].long()
        p_scores = p["scores"].float()
        t_boxes = t["boxes"].float()
        t_labels = t["labels"].long()

        classes = torch.cat([p_labels, t_labels]).unique()
        for c in classes.tolist():
            det_mask = (p_labels == c)
            gt_mask  = (t_labels == c)
            det_count = int(det_mask.sum().item())
            gt_count = int(gt_mask.sum().item())

            if det_count == 0:
                all_ious.extend([0.0] * gt_count)
                continue
            if gt_count == 0:
                all_ious.extend([0.0] * det_count)
                continue

            d_boxes = p_boxes[det_mask]
            d_scores = p_scores[det_mask]
            g_boxes = t_boxes[gt_mask]

            order = torch.argsort(d_scores, descending=True) # sort detections by confidence desc 
            d_boxes = d_boxes[order]
            ious = box_iou(d_boxes, g_boxes) 
            gt_taken = torch.zeros(g_boxes.size(0), dtype=torch.bool, device=ious.device)

            for di in range(ious.size(0)):
                iou_row = ious[di].masked_fill(gt_taken, -1.0) # ignore already taken GT
                best_iou, best_j = torch.max(iou_row, dim=0)
                if best_iou > 0.0: #TP
                    all_ious.append(best_iou.item())
                    gt_taken[best_j] = True 
                else:
                    #FP asigned to 0 IoU
                    all_ious.append(0.0)

            #FN asigned to 0 IoU
            fn_count = int((~gt_taken).sum().item())
            if fn_count:
                all_ious.extend([0.0] * fn_count)
    if len(all_ious) == 0:
        return 1.0
    return float(sum(all_ious) / len(all_ious))


def get_metrics(detections: list[dict], annotations: list[dict], metrics: list | None) -> dict:
    if metrics is None:
        metrics = ["iou"]
    allowed = {"lrp", "iou"}
    assert all(m in allowed for m in metrics), f"Unsupported metric(s): {set(metrics) - allowed}"

    results = {}

    if "iou" in metrics:
        iou_out = matched_iou(detections, annotations)
        results["iou"] = iou_out
    
    if "lrp" in metrics:
        lrp_metric = LocalizationRecallPrecision(tau=0.5, box_format="xyxy")
        lrp_metric.update(detections, annotations)
        lrp_out = lrp_metric.compute()
        results["lrp"] = 1.0 - float(lrp_out)  # higher is better

    return results


if __name__ == "__main__":
    detections = [
        {
            "boxes": torch.tensor([[1, 1, 2, 2]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
            "scores": torch.tensor([0.9]),
        },
        {
            "boxes": torch.tensor([[10, 10, 12, 12]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
            "scores": torch.tensor([0.8]),
        },
    ]
    annotations = [
        {
            "boxes": torch.tensor([[1, 1, 2, 2]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
        },
        {
            "boxes": torch.tensor([[5, 5, 7, 7]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
        },
    ]

    metrics = get_metrics(detections, annotations, metrics=["iou", "lrp"])
    print(metrics)
