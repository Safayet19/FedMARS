from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_layers: list[str]
    selected_bits: int
    budget_bits: int
    model_bits: int
    budget_fraction: float
    budget_violation: bool


def select_layers_under_budget(
    global_credit: Mapping[str, float],
    layer_bits: Mapping[str, int],
    budget_fraction: float,
    threshold: float,
    ensure_nonempty: bool = True,
) -> list[str]:
    return select_layers_with_report(global_credit, layer_bits, budget_fraction, threshold, ensure_nonempty).selected_layers


def select_layers_with_report(
    global_credit: Mapping[str, float],
    layer_bits: Mapping[str, int],
    budget_fraction: float,
    threshold: float,
    ensure_nonempty: bool = True,
) -> SelectionResult:
    names = [n for n in layer_bits.keys() if n in global_credit and int(layer_bits[n]) > 0]
    model_bits = int(sum(int(layer_bits[n]) for n in names))
    budget_fraction = float(np.clip(float(budget_fraction), 0.0, 1.0))
    budget_bits = int(np.floor(model_bits * budget_fraction + 1e-9))
    if model_bits <= 0:
        return SelectionResult([], 0, 0, 0, budget_fraction, False)
    if budget_fraction >= 0.999999:
        selected = list(names)
        bits = int(sum(int(layer_bits[n]) for n in selected))
        return SelectionResult(selected, bits, budget_bits, model_bits, budget_fraction, bits > budget_bits)
    candidates = [n for n in names if float(global_credit.get(n, 0.0)) >= float(threshold)]
    ordered = sorted(
        candidates,
        key=lambda n: (float(global_credit[n]) / max(int(layer_bits[n]), 1), float(global_credit[n])),
        reverse=True,
    )
    selected: list[str] = []
    used = 0
    for name in ordered:
        bits = int(layer_bits[name])
        if used + bits <= budget_bits:
            selected.append(name)
            used += bits
    if not selected and ensure_nonempty:
        feasible = [n for n in candidates if int(layer_bits[n]) <= budget_bits]
        if feasible:
            best = max(feasible, key=lambda n: (float(global_credit[n]), -int(layer_bits[n])))
            selected = [best]
            used = int(layer_bits[best])
    selected = [n for n in names if n in set(selected)]
    selected_bits = int(sum(int(layer_bits[n]) for n in selected))
    return SelectionResult(selected, selected_bits, budget_bits, model_bits, budget_fraction, selected_bits > budget_bits)


def _credit_weight(layer_name: str, idx: int, credit_dicts: Sequence[Mapping[str, float]], gamma: float) -> float:
    vals = np.asarray([float(d.get(layer_name, 0.0)) for d in credit_dicts], dtype=float)
    if len(vals) == 0:
        return 1.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-8
    z = float(np.clip((vals[idx] - med) / mad, -2.5, 2.5))
    return 1.0 + float(gamma) * max(0.0, z) / 2.5


def aggregate_sparse_updates(
    sparse_updates: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
    client_weights: Sequence[float],
    selected_layers: Sequence[str],
    client_credit_dicts: Sequence[Mapping[str, float]],
    credit_weight_gamma: float = 1.0,
    delta_clip_factor: float = 2.5,
) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_name in selected_layers:
        layer_payloads: list[Mapping[str, torch.Tensor]] = []
        layer_weights: list[float] = []
        for idx, update in enumerate(sparse_updates):
            if layer_name not in update:
                continue
            weight = float(client_weights[idx]) * _credit_weight(layer_name, idx, client_credit_dicts, credit_weight_gamma)
            if weight <= 0.0:
                continue
            layer_payloads.append(update[layer_name])
            layer_weights.append(weight)
        if sum(layer_weights) <= 0.0:
            continue
        layer_accum: dict[str, torch.Tensor] = {}
        param_names = set().union(*(p.keys() for p in layer_payloads)) if layer_payloads else set()
        for pname in param_names:
            deltas: list[torch.Tensor] = []
            weights: list[float] = []
            for payload, weight in zip(layer_payloads, layer_weights):
                if pname in payload:
                    deltas.append(payload[pname])
                    weights.append(weight)
            if not deltas:
                continue
            norms = [float(torch.norm(delta)) for delta in deltas]
            median_norm = float(np.median(np.asarray(norms, dtype=float))) if norms else 0.0
            clip_threshold = float(delta_clip_factor) * median_norm if median_norm > 0 else (max(norms) if norms else 0.0)
            acc = torch.zeros_like(deltas[0])
            denom = 0.0
            for delta, weight, norm in zip(deltas, weights, norms):
                if clip_threshold > 0.0 and norm > clip_threshold:
                    delta = delta * (clip_threshold / (norm + 1e-12))
                acc += float(weight) * delta
                denom += float(weight)
            if denom > 0.0:
                layer_accum[pname] = acc / denom
        if layer_accum:
            out[layer_name] = layer_accum
    return out


def apply_global_update(
    model: torch.nn.Module,
    aggregated_updates: Mapping[str, Mapping[str, torch.Tensor]],
    layer_steps: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for layer_name, layer_update in aggregated_updates.items():
            step = float(layer_steps.get(layer_name, 1.0))
            for pname, delta in layer_update.items():
                named_params[pname].add_(step * delta.to(named_params[pname].device))
    return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
