from __future__ import annotations

import numpy as np
import torch

from .utils import safe_cosine


def select_counterfactual_mixture(
    gradients: list[torch.Tensor],
    reference: torch.Tensor | None,
    beta: float,
    temperature: float,
) -> tuple[np.ndarray, torch.Tensor, float, float]:
    if len(gradients) == 0:
        raise ValueError("At least one gradient is required.")
    if len(gradients) == 1:
        only = gradients[0].detach().clone()
        return np.array([1.0], dtype=float), only, 0.0, float(torch.norm(only))

    J = len(gradients)
    pair_conflicts = np.zeros((J, J), dtype=np.float32)
    for i in range(J):
        for j in range(J):
            if i != j:
                pair_conflicts[i, j] = 1.0 - safe_cosine(gradients[i], gradients[j])

    cluster_scores = np.zeros(J, dtype=np.float32)
    for j in range(J):
        if reference is None or float(torch.norm(reference)) <= 1e-12:
            align = float(torch.norm(gradients[j]))
        else:
            align = safe_cosine(reference, gradients[j])
        cluster_scores[j] = align - beta * float(pair_conflicts[j].mean())

    scaled = (cluster_scores - cluster_scores.max()) / max(float(temperature), 1e-6)
    weights = np.exp(scaled)
    weights = weights / max(float(weights.sum()), 1e-12)

    mixed = sum(float(weights[j]) * gradients[j] for j in range(J))

    conflict = 0.0
    for i in range(J):
        for j in range(i + 1, J):
            conflict += float(weights[i] * weights[j]) * (1.0 - safe_cosine(gradients[i], gradients[j]))

    objective = float(np.dot(cluster_scores, weights) - beta * conflict)
    return weights, mixed.detach().clone(), float(conflict), objective