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
    norm: float
    probe_gain: float


def build_reference_sketch(previous_layer_vector: torch.Tensor | None, mode: str = "unit") -> torch.Tensor | None:
    if previous_layer_vector is None:
        return None
    vec = previous_layer_vector.detach().reshape(-1).float().clone()
    norm = float(torch.norm(vec))
    if norm <= 1e-12:
        return None
    if mode in {"unit", "ema_unit"}:
        return vec / norm
    if mode == "sign":
        return torch.sign(vec)
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
    if reference is None or float(torch.norm(reference)) <= 1e-12:
        alignment = 1.0 if grad_norm > 0 else 0.0
    else:
        alignment = safe_cosine(reference, mixed_gradient)

    benefit = float(alignment * np.log1p(grad_norm))
    risk = float(residual_conflict)
    probe = float(probe_gain)

    credit = float(
        depth_weight
        * (benefit - float(lambda_r) * risk - float(lambda_c) * float(cost) + float(lambda_v) * probe)
    )

    return LayerCreditRecord(
        benefit=benefit,
        risk=risk,
        cost=float(cost),
        depth_weight=float(depth_weight),
        credit=credit,
        alignment=float(alignment),
        norm=float(grad_norm),
        probe_gain=probe,
    )


def aggregate_global_credit(client_credit_dicts: list[Mapping[str, float]], layer_names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in layer_names:
        vals = np.asarray([float(d.get(name, 0.0)) for d in client_credit_dicts], dtype=float)
        if len(vals) == 0:
            out[name] = 0.0
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) + 1e-8
        clipped = np.clip(vals, med - 2.5 * mad, med + 2.5 * mad)
        out[name] = float(np.mean(clipped))
    return out