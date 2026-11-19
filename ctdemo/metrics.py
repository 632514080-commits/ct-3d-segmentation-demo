from __future__ import annotations

import numpy as np


def segmentation_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred, target = pred.astype(bool), target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    eps = 1e-8
    return {
        "dice": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
        "iou": float((tp + eps) / (tp + fp + fn + eps)),
        "sensitivity": float((tp + eps) / (tp + fn + eps)),
        "predicted_voxels": int(pred.sum()),
        "target_voxels": int(target.sum()),
    }

