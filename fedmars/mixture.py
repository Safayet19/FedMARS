from __future__ import annotations

import itertools

import numpy as np
import torch

from .utils import safe_cosine


def simplex_grid(num_components: int, resolution: int) -> list[np.ndarray]:
    if num_components <= 0:
        raise ValueError("num_components must be positive.")
    if num_components == 1:
        return [np.array([1.0], dtype=float)]
    if resolution <= 1:
        out = []
        for idx in range(num_components):
            vec = np.zeros(num_components, dtype=float)
            vec[idx] = 1.0
            out.append(vec)
        return out
    points: list[np.ndarray] = []
    for counts in itertools.product(range(resolution + 1), repeat=num_components):
        if sum(counts) == resolution:
            points.append(np.array(counts, dtype=float) / resolution)
    return points or [np.ones(num_components, dtype=float) / num_components]


def mixed_gradient(gradients: list[torch.Tensor], weights: np.ndarray) -> torch.Tensor:
    out = torch.zeros_like(gradients[0])
    for weight, grad in zip(weights, gradients):
        out = out + float(weight) * grad
    return out


def conflict_penalty(gradients: list[torch.Tensor], weights: np.ndarray) -> float:
    if len(gradients) <= 1:
        return 0.0
    total = 0.0
    for i in range(len(gradients)):
        for j in range(i + 1, len(gradients)):
            total += float(weights[i] * weights[j]) * (1.0 - safe_cosine(gradients[i], gradients[j]))
    return float(total)


def select_counterfactual_mixture(
    gradients: list[torch.Tensor],
    reference: torch.Tensor | None,
    beta: float,
    resolution: int,
) -> tuple[np.ndarray, torch.Tensor, float, float]:
    if len(gradients) == 0:
        raise ValueError("At least one gradient is required.")
    if len(gradients) == 1:
        only = gradients[0].detach().clone()
        return np.array([1.0], dtype=float), only, 0.0, float(torch.norm(only))

    best_weights = None
    best_mixed = None
    best_conflict = None
    best_score = -float("inf")
    for weights in simplex_grid(len(gradients), resolution):
        mixed = mixed_gradient(gradients, weights)
        conflict = conflict_penalty(gradients, weights)
        if reference is None or float(torch.norm(reference)) <= 1e-12:
            score = float(torch.norm(mixed)) - beta * conflict
        else:
            score = float(torch.dot(reference.reshape(-1).float(), mixed.reshape(-1).float()) - beta * conflict)
        if score > best_score:
            best_score = score
            best_weights = weights.copy()
            best_mixed = mixed.detach().clone()
            best_conflict = float(conflict)
    assert best_weights is not None and best_mixed is not None and best_conflict is not None
    return best_weights, best_mixed, best_conflict, float(best_score)
