"""Evaluation metrics tailored for zeta forecasting."""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional

import torch


MetricFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def bias(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(pred - target)


def nrmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.sqrt(torch.mean((pred - target) ** 2))
    denom = torch.max(target) - torch.min(target)
    return numerator / torch.clamp(denom, min=1e-6)


def pearson_corr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean
    numerator = (pred_centered * target_centered).sum(dim=1)
    denom = torch.sqrt(
        torch.sum(pred_centered ** 2, dim=1) * torch.sum(target_centered ** 2, dim=1)
    )
    corr = numerator / torch.clamp(denom, min=1e-6)
    return corr.mean()


DEFAULT_METRICS: Mapping[str, MetricFn] = {
    "rmse": rmse,
    "mae": mae,
    "bias": bias,
    "nrmse": nrmse,
    "pearson_corr": pearson_corr,
}


def evaluate_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    metrics: Optional[Mapping[str, MetricFn]] = None,
) -> Dict[str, float]:
    selected = metrics or DEFAULT_METRICS
    return {name: float(func(pred, target).detach().cpu()) for name, func in selected.items()}
