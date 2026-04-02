from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

def select_layers_under_budget(
    global_credit: Mapping[str, float],
    layer_costs: Mapping[str, float],
    budget_fraction: float,
    threshold: float,
    budget_scale: int = 200,
    ensure_nonempty: bool = True,
) -> list[str]:
    candidates = [
        name for name, credit in global_credit.items()
        if credit >= threshold and layer_costs.get(name, 0.0) > 0.0
    ]
    if not candidates:
        if not ensure_nonempty or not global_credit:
            return []
        return [max(global_credit.items(), key=lambda kv: kv[1])[0]]

    costs = {name: max(1, int(round(layer_costs[name] * budget_scale))) for name in candidates}
    values = {name: max(0.0, float(global_credit[name])) for name in candidates}
    capacity = max(1, int(round(float(budget_fraction) * budget_scale)))

    dp = np.zeros((len(candidates) + 1, capacity + 1), dtype=float)
    keep = np.zeros((len(candidates) + 1, capacity + 1), dtype=bool)
    for i, name in enumerate(candidates, start=1):
        cost = costs[name]
        value = values[name]
        for c in range(capacity + 1):
            dp[i, c] = dp[i - 1, c]
            if cost <= c:
                alt = dp[i - 1, c - cost] + value
                if alt > dp[i, c]:
                    dp[i, c] = alt
                    keep[i, c] = True

    selected: list[str] = []
    c = capacity
    for i in range(len(candidates), 0, -1):
        if keep[i, c]:
            name = candidates[i - 1]
            selected.append(name)
            c -= costs[name]
    selected = sorted(selected)
    if not selected and ensure_nonempty:
        selected = [max(candidates, key=lambda name: global_credit[name])]
    return selected

def aggregate_sparse_updates(
    sparse_updates: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
    client_weights: Sequence[float],
    selected_layers: Sequence[str],
) -> dict[str, dict[str, torch.Tensor]]:
    total_weight = float(sum(client_weights)) or 1.0
    out: dict[str, dict[str, torch.Tensor]] = {}
    for layer_name in selected_layers:
        layer_accum: dict[str, torch.Tensor] = {}
        any_seen = False
        for update, weight in zip(sparse_updates, client_weights):
            if layer_name not in update:
                continue
            any_seen = True
            for pname, delta in update[layer_name].items():
                if pname not in layer_accum:
                    layer_accum[pname] = torch.zeros_like(delta)
                layer_accum[pname] += (float(weight) / total_weight) * delta
        if any_seen:
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
    return {k: v.detach().clone() for k, v in model.state_dict().items()}
