from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True, slots=True)
class LayerSpec:
    name: str
    param_names: tuple[str, ...]
    depth_index: int
    numel: int


def _layer_group_name(param_name: str) -> str:
    if "." not in param_name:
        return param_name
    head, tail = param_name.rsplit(".", 1)
    if tail in {"weight", "bias"}:
        return head
    return param_name


def build_layer_specs(model: torch.nn.Module) -> list[LayerSpec]:
    groups: dict[str, list[str]] = {}
    for name, _ in model.named_parameters():
        groups.setdefault(_layer_group_name(name), []).append(name)
    specs: list[LayerSpec] = []
    named_params = dict(model.named_parameters())
    for depth_idx, key in enumerate(groups.keys(), start=1):
        numel = sum(int(named_params[name].numel()) for name in groups[key])
        specs.append(LayerSpec(name=key, param_names=tuple(groups[key]), depth_index=depth_idx, numel=numel))
    return specs


def layer_name_to_spec(layer_specs: list[LayerSpec]) -> dict[str, LayerSpec]:
    return {spec.name: spec for spec in layer_specs}


def compute_depth_weights(layer_specs: list[LayerSpec], mode: str = "linear") -> dict[str, float]:
    L = max(len(layer_specs), 1)
    out: dict[str, float] = {}
    for spec in layer_specs:
        if mode == "linear":
            out[spec.name] = spec.depth_index / L
        elif mode == "uniform":
            out[spec.name] = 1.0
        else:
            raise ValueError(f"Unsupported depth_weight_mode: {mode}")
    return out


def compute_layer_costs(layer_specs: list[LayerSpec]) -> dict[str, float]:
    total = float(sum(spec.numel for spec in layer_specs))
    if total <= 0:
        return {spec.name: 0.0 for spec in layer_specs}
    return {spec.name: float(spec.numel / total) for spec in layer_specs}


def flatten_params_from_state(state_dict: Mapping[str, torch.Tensor], spec: LayerSpec) -> torch.Tensor:
    flat = [state_dict[name].reshape(-1).float() for name in spec.param_names]
    return torch.cat(flat, dim=0) if flat else torch.zeros(1)


def flatten_grads_from_model(model: torch.nn.Module, spec: LayerSpec) -> torch.Tensor:
    named_params = dict(model.named_parameters())
    chunks: list[torch.Tensor] = []
    for name in spec.param_names:
        grad = named_params[name].grad
        if grad is None:
            chunks.append(torch.zeros_like(named_params[name]).reshape(-1).float())
        else:
            chunks.append(grad.detach().reshape(-1).float())
    return torch.cat(chunks, dim=0) if chunks else torch.zeros(1)


def state_delta_by_layer(
    new_state: Mapping[str, torch.Tensor],
    old_state: Mapping[str, torch.Tensor],
    layer_specs: list[LayerSpec],
) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for spec in layer_specs:
        layer_delta: dict[str, torch.Tensor] = {}
        for pname in spec.param_names:
            layer_delta[pname] = new_state[pname] - old_state[pname]
        out[spec.name] = layer_delta
    return out
