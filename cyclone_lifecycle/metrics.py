"""Reusable metric calculations and streaming accumulators."""

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


UNIT_SCALE_FOR_STD = {
    "msl": 1.0 / 100.0,
}


def compute_row_metrics(
    target_frame: np.ndarray,
    pred_frame: np.ndarray,
    box: Optional[Tuple[int, int, int, int]] = None,
    dataset_std: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Compute MAE, RMSE, bias, and optionally normalized RMSE for one field."""
    if box is not None:
        r0, r1, c0, c1 = box
        target_frame = target_frame[r0:r1, c0:c1]
        pred_frame = pred_frame[r0:r1, c0:c1]

    error = pred_frame - target_frame
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": rmse,
        "bias": float(np.mean(error)),
        "nrmse": rmse / dataset_std if dataset_std is not None and dataset_std > 1e-8 else None,
    }


def get_dataset_std(
    var_name: str,
    norm_stats: dict,
    label: str = "target",
    unit_scale: float = 1.0,
) -> Optional[float]:
    """Read a variable's standard deviation from flat or model-keyed statistics."""
    if not isinstance(norm_stats, dict):
        return None

    if label in norm_stats and isinstance(norm_stats[label], dict):
        stats_dict = norm_stats[label]
    elif "target" in norm_stats and isinstance(norm_stats["target"], dict):
        stats_dict = norm_stats["target"]
    else:
        stats_dict = norm_stats

    var_stats = stats_dict.get(var_name)
    if isinstance(var_stats, dict):
        std = var_stats.get("std")
        return std * unit_scale if std is not None else None
    return None


class MaskedMetrics:
    """Streaming MSE, MAE, and bias over masked entries in normalized units."""

    def __init__(self, variables: Sequence[str], mode_names: Sequence[str]):
        self.variables = variables
        self.mode_names = mode_names
        self.reset()

    def reset(self):
        channel_count, mode_count = len(self.variables), len(self.mode_names)
        zeros = lambda count: torch.zeros(count, dtype=torch.float64)
        self.sq, self.ab, self.bi, self.n = (
            zeros(channel_count), zeros(channel_count), zeros(channel_count), zeros(channel_count)
        )
        self.m_sq, self.m_n = zeros(mode_count), zeros(mode_count)

    @torch.no_grad()
    def update(self, preds, targets, mask, modes):
        error = (preds - targets).double()
        missing = mask.double()
        self.sq += ((error ** 2) * missing).sum(dim=(0, 2, 3)).cpu()
        self.ab += (error.abs() * missing).sum(dim=(0, 2, 3)).cpu()
        self.bi += (error * missing).sum(dim=(0, 2, 3)).cpu()
        self.n += missing.sum(dim=(0, 2, 3)).cpu()
        modes = modes.cpu()
        self.m_sq.index_add_(0, modes, ((error ** 2) * missing).sum(dim=(1, 2, 3)).cpu())
        self.m_n.index_add_(0, modes, missing.sum(dim=(1, 2, 3)).cpu())

    def calculate_metrics(self):
        nan = float("nan")
        n_total = self.n.sum().clamp(min=1)
        summary = {
            "overall": {
                "mse": float(self.sq.sum() / n_total),
                "mae": float(self.ab.sum() / n_total),
                "bias": float(self.bi.sum() / n_total),
            },
            "per_variable": {},
            "per_mode": {},
        }
        for index, variable in enumerate(self.variables):
            count = float(self.n[index])
            summary["per_variable"][variable] = {
                "mse": float(self.sq[index]) / count if count > 0 else nan,
                "mae": float(self.ab[index]) / count if count > 0 else nan,
                "bias": float(self.bi[index]) / count if count > 0 else nan,
            }
        for index, name in enumerate(self.mode_names):
            count = float(self.m_n[index])
            summary["per_mode"][name] = float(self.m_sq[index]) / count if count > 0 else nan
        return summary


class MaskedRecoveryAccumulator:
    """Streaming metrics for masked-VAE evaluation, including derived wind speed."""

    def __init__(self, channel_count: int):
        zeros = lambda: np.zeros(channel_count, dtype=np.float64)
        self.sq, self.ab, self.bi, self.tsq, self.n = zeros(), zeros(), zeros(), zeros(), zeros()
        self.w = np.zeros(4)

    @torch.no_grad()
    def add(self, pred, target, mask, mean, std, uv):
        error = (pred - target).double()
        target = target.double()
        missing = mask.double()
        reduce = lambda values: values.sum(dim=(0, 2, 3)).cpu().numpy()
        self.sq += reduce(error ** 2 * missing)
        self.ab += reduce(error.abs() * missing)
        self.bi += reduce(error * missing)
        self.tsq += reduce(target ** 2 * missing)
        self.n += reduce(missing)
        if uv is not None:
            iu, iv = uv
            wind_mask = mask[:, iu] & mask[:, iv]
            if wind_mask.any():
                pred_u = pred[:, iu].double() * std[iu] + mean[iu]
                pred_v = pred[:, iv].double() * std[iv] + mean[iv]
                target_u = target[:, iu].double() * std[iu] + mean[iu]
                target_v = target[:, iv].double() * std[iv] + mean[iv]
                wind_error = (
                    torch.hypot(pred_u, pred_v) - torch.hypot(target_u, target_v)
                )[wind_mask]
                self.w += np.array([
                    (wind_error ** 2).sum().item(),
                    wind_error.abs().sum().item(),
                    wind_error.sum().item(),
                    float(wind_mask.sum()),
                ])


def summarize_masked_recovery(acc, vars_list, std_np):
    results = {"per_variable": {}, "wind_speed": None}
    for index, variable in enumerate(vars_list):
        count = acc.n[index]
        if count == 0:
            continue
        mse = acc.sq[index] / count
        results["per_variable"][variable] = {
            "n": int(count),
            "rmse_norm": float(math.sqrt(mse)),
            "mae_norm": float(acc.ab[index] / count),
            "bias_norm": float(acc.bi[index] / count),
            "skill": float(1.0 - acc.sq[index] / max(acc.tsq[index], 1e-12)),
            "rmse_phys": float(math.sqrt(mse) * std_np[index]),
            "mae_phys": float(acc.ab[index] / count * std_np[index]),
            "bias_phys": float(acc.bi[index] / count * std_np[index]),
        }
    count_total = acc.n.sum()
    results["overall_rmse_norm"] = (
        float(math.sqrt(acc.sq.sum() / count_total)) if count_total > 0 else float("nan")
    )
    if acc.w[3] > 0:
        results["wind_speed"] = {
            "n": int(acc.w[3]),
            "rmse_phys": float(math.sqrt(acc.w[0] / acc.w[3])),
            "mae_phys": float(acc.w[1] / acc.w[3]),
            "bias_phys": float(acc.w[2] / acc.w[3]),
        }
    return results


class DiffusionErrorAccumulator:
    """Streaming per-variable errors and spatial sharpness for diffusion evaluation."""

    def __init__(self, channel_count: int):
        self.sq = np.zeros(channel_count)
        self.ab = np.zeros(channel_count)
        self.bi = np.zeros(channel_count)
        self.gp = np.zeros(channel_count)
        self.gt = np.zeros(channel_count)
        self.n = 0.0

    @staticmethod
    def _grad_mag(values: torch.Tensor) -> torch.Tensor:
        dx = values[..., :-1, 1:] - values[..., :-1, :-1]
        dy = values[..., 1:, :-1] - values[..., :-1, :-1]
        return (dx ** 2 + dy ** 2).sqrt()

    def add(self, error: torch.Tensor, pred: Optional[torch.Tensor] = None,
            truth: Optional[torch.Tensor] = None):
        if pred is not None:
            if truth is None:
                raise ValueError("truth is required when computing diffusion sharpness")
            self.gp += self._grad_mag(pred.double()).sum(dim=(0, 1, 3, 4)).cpu().numpy()
            self.gt += self._grad_mag(truth.double()).sum(dim=(0, 1, 3, 4)).cpu().numpy()
        values = error.double()
        self.sq += (values ** 2).sum(dim=(0, 1, 3, 4)).cpu().numpy()
        self.ab += values.abs().sum(dim=(0, 1, 3, 4)).cpu().numpy()
        self.bi += values.sum(dim=(0, 1, 3, 4)).cpu().numpy()
        self.n += values.shape[0] * values.shape[1] * values.shape[3] * values.shape[4]

    def summary(self, vars_list, phys_factor):
        per_variable = {}
        for index, variable in enumerate(vars_list):
            rmse_norm = float(np.sqrt(self.sq[index] / self.n))
            per_variable[variable] = {
                "rmse": rmse_norm * phys_factor[index],
                "mae": float(self.ab[index] / self.n) * phys_factor[index],
                "bias": float(self.bi[index] / self.n) * phys_factor[index],
                "rmse_norm": rmse_norm,
                "sharpness": float(self.gp[index] / self.gt[index]) if self.gt[index] > 0 else float("nan"),
            }
        return {
            "per_variable": per_variable,
            "overall_rmse_norm": float(np.sqrt(self.sq.sum() / (self.n * len(vars_list)))),
        }
