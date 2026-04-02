from __future__ import annotations

from typing import Mapping, Sequence

import torch


def select_layers_under_budget(
    global_credit: Mapping[str, float],
    layer_costs: Mapping[str, float],
    budget_fraction: float,
    threshold: float,
    budget_scale: int = 200,
    ensure_nonempty: bool = True,
) -> list[str]:
    candidates = [name for name, credit in global_credit.items() if credit >= threshold]
    if not candidates:
        if not ensure_nonempty or not global_credit:
            return []
        return [max(global_credit.items(), key=lambda kv: kv[1])[0]]

    ordered = sorted(
        candidates,
        key=lambda name: (
            float(global_credit[name]) / max(float(layer_costs.get(name, 1e-12)), 1e-12),
            float(global_credit[name]),
        ),
        reverse=True,
    )

    selected: list[str] = []
    used = 0.0
    cap = float(budget_fraction)

    for name in ordered:
        cost = float(layer_costs.get(name, 0.0))
        if used + cost <= cap + 1e-12:
            selected.append(name)
            used += cost

    if not selected and ensure_nonempty:
        selected = [max(candidates, key=lambda name: global_credit[name])]
    return sorted(selected)


def aggregate_sparse_updates(
    sparse_updates: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
    client_weights: Sequence[float],
    selected_layers: Sequence[str],
    client_credit_dicts: Sequence[Mapping[str, float]] | None = None,
    use_credit_weighting: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}

    for layer_name in selected_layers:
        layer_accum: dict[str, torch.Tensor] = {}
        layer_weights: list[float] = []
        layer_payloads: list[Mapping[str, torch.Tensor]] = []

        for idx, update in enumerate(sparse_updates):
            if layer_name not in update:
                continue
            base_weight = float(client_weights[idx])
            if use_credit_weighting and client_credit_dicts is not None:
                trust = 1.0 + 2.0 * max(0.0, float(client_credit_dicts[idx].get(layer_name, 0.0)))
                base_weight *= trust
            layer_payloads.append(update[layer_name])
            layer_weights.append(base_weight)

        total_weight = float(sum(layer_weights))
        if total_weight <= 0.0:
            continue

        for payload, weight in zip(layer_payloads, layer_weights):
            for pname, delta in payload.items():
                if pname not in layer_accum:
                    layer_accum[pname] = torch.zeros_like(delta)
                layer_accum[pname] += (float(weight) / total_weight) * delta

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