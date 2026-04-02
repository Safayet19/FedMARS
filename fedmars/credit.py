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


def build_reference_sketch(previous_layer_vector: torch.Tensor | None, mode: str = "unit") -> torch.Tensor | None:
    if previous_layer_vector is None:
        return None
    vec = previous_layer_vector.detach().reshape(-1).float().clone()
    norm = float(torch.norm(vec))
    if norm <= 1e-12:
        return None
    if mode == "unit":
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
) -> LayerCreditRecord:
    grad_norm = float(torch.norm(mixed_gradient))
    if reference is None or float(torch.norm(reference)) <= 1e-12:
        alignment = 1.0 if grad_norm > 0 else 0.0
        benefit = grad_norm
    else:
        alignment = safe_cosine(reference, mixed_gradient)
        benefit = alignment * grad_norm
    risk = float(residual_conflict)
    credit = float(depth_weight * (benefit - lambda_r * risk - lambda_c * cost))
    return LayerCreditRecord(
        benefit=benefit,
        risk=risk,
        cost=float(cost),
        depth_weight=float(depth_weight),
        credit=credit,
        alignment=float(alignment),
        norm=grad_norm,
    )


def aggregate_global_credit(client_credit_dicts: list[Mapping[str, float]], layer_names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in layer_names:
        vals = [float(d.get(name, 0.0)) for d in client_credit_dicts]
        out[name] = float(np.median(np.asarray(vals, dtype=float))) if vals else 0.0
    return out
