from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from .utils import safe_cosine


@dataclass(slots=True)
class LayerCreditRecord:
    benefit: float
    risk: float
    cost: float
    depth_weight: float
    credit: float
    alignment: float
    gradient_norm: float
    probe_gain: float


def build_reference_sketch(
    previous_layer_vector: torch.Tensor | None,
    mode: str = "unit",
    topk_fraction: float = 0.10,
) -> torch.Tensor | None:
    if previous_layer_vector is None:
        return None
    vec = previous_layer_vector.detach().reshape(-1).float().clone()
    norm = float(torch.norm(vec))
    if norm <= 1e-12:
        return None
    base_mode = mode[4:] if mode.startswith("ema_") else mode
    if base_mode == "unit":
        return vec / norm
    if base_mode == "sign":
        signed = torch.sign(vec)
        signed_norm = float(torch.norm(signed))
        if signed_norm <= 1e-12:
            return None
        return signed / signed_norm
    if base_mode == "topk_sign":
        frac = min(max(float(topk_fraction), 0.0), 1.0)
        k = max(1, int(np.ceil(frac * vec.numel())))
        idx = torch.topk(torch.abs(vec), k=k, largest=True).indices
        out = torch.zeros_like(vec)
        out[idx] = torch.sign(vec[idx])
        out_norm = float(torch.norm(out))
        if out_norm <= 1e-12:
            return None
        return out / out_norm
    raise ValueError(f"Unsupported reference_sketch_mode: {mode}")


def compute_layer_credit(
    reference: torch.Tensor | None,
    mixed_gradient: torch.Tensor,
    residual_conflict: float,
    cost: float,
    depth_weight: float,
    lambda_r: float,
    lambda_c: float,
    probe_gain: float = 0.0,
    lambda_v: float = 0.0,
) -> LayerCreditRecord:
    grad_norm = float(torch.norm(mixed_gradient))
    if grad_norm <= 1e-12:
        alignment = 0.0
        benefit = 0.0
    elif reference is None or float(torch.norm(reference)) <= 1e-12:
        alignment = 1.0
        benefit = grad_norm
    else:
        alignment = safe_cosine(reference, mixed_gradient)
        benefit = float(alignment * grad_norm)
    risk = float(residual_conflict)
    probe_term = float(np.tanh(5.0 * float(probe_gain)))
    credit = float(depth_weight * (benefit - float(lambda_r) * risk - float(lambda_c) * float(cost) + float(lambda_v) * probe_term))
    return LayerCreditRecord(
        benefit=float(benefit),
        risk=risk,
        cost=float(cost),
        depth_weight=float(depth_weight),
        credit=credit,
        alignment=float(alignment),
        gradient_norm=float(grad_norm),
        probe_gain=float(np.tanh(5.0 * float(probe_gain))),
    )


def aggregate_global_credit(
    client_credit_dicts: list[Mapping[str, float]],
    layer_names: list[str],
    method: str = "median",
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in layer_names:
        vals = np.asarray([float(d.get(name, 0.0)) for d in client_credit_dicts], dtype=float)
        if len(vals) == 0:
            out[name] = 0.0
            continue
        if method == "median":
            out[name] = float(np.median(vals))
            continue
        if method == "trimmed_mean":
            if len(vals) <= 2:
                out[name] = float(np.mean(vals))
            else:
                trim = max(1, int(0.2 * len(vals)))
                trimmed = np.sort(vals)[trim:-trim] if len(vals) > 2 * trim else vals
                out[name] = float(np.mean(trimmed))
            continue
        if method == "clipped_mean":
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med))) + 1e-8
            clipped = np.clip(vals, med - 2.5 * mad, med + 2.5 * mad)
            out[name] = float(np.mean(clipped))
            continue
        raise ValueError(f"Unsupported global credit aggregation method: {method}")
    return out


def postprocess_control_credit(
    raw_credit: Mapping[str, float],
    mode: str = "none",
    clip: float = 2.5,
) -> dict[str, float]:
    if mode == "none":
        return {name: float(value) for name, value in raw_credit.items()}
    if mode != "robust_zscore":
        raise ValueError(f"Unsupported control_credit_mode: {mode}")
    if not raw_credit:
        return {}
    vals = np.asarray(list(raw_credit.values()), dtype=float)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-8
    scale = 1.4826 * mad + 1e-8
    out: dict[str, float] = {}
    for name, value in raw_credit.items():
        z = (float(value) - med) / scale
        out[name] = float(np.clip(z, -abs(float(clip)), abs(float(clip))))
    return out
