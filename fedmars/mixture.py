from __future__ import annotations

import numpy as np
import torch

from .utils import safe_cosine


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    if v.ndim != 1:
        raise ValueError("Simplex projection expects a 1D vector.")
    n = v.shape[0]
    if n == 1:
        return np.array([1.0], dtype=np.float32)
    clean = np.nan_to_num(v.astype(np.float64, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    u = np.sort(clean)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full(n, 1.0 / n, dtype=np.float32)
    rho = ind[cond][-1]
    theta = cssv[rho - 1] / rho
    w = np.maximum(clean - theta, 0.0)
    s = float(w.sum())
    if s <= 0.0 or not np.isfinite(s):
        return np.full(n, 1.0 / n, dtype=np.float32)
    return (w / s).astype(np.float32)


def select_counterfactual_mixture(
    gradients: list[torch.Tensor],
    reference: torch.Tensor | None,
    beta: float,
    temperature: float,
    entropy_coef: float = 0.048,
    steps: int = 40,
) -> tuple[np.ndarray, torch.Tensor, float, float]:
    if len(gradients) == 0:
        raise ValueError("At least one gradient is required.")
    clean_gradients = [torch.nan_to_num(g.detach().clone().float(), nan=0.0, posinf=0.0, neginf=0.0) for g in gradients]
    if len(clean_gradients) == 1:
        only = clean_gradients[0]
        return np.array([1.0], dtype=np.float32), only, 0.0, float(torch.norm(only))
    J = len(clean_gradients)
    align_scores = np.zeros(J, dtype=np.float32)
    conflict_matrix = np.zeros((J, J), dtype=np.float32)
    ref = None if reference is None else torch.nan_to_num(reference.detach().clone().float(), nan=0.0, posinf=0.0, neginf=0.0)
    for i, grad in enumerate(clean_gradients):
        if ref is None or float(torch.norm(ref)) <= 1e-12:
            align_scores[i] = float(torch.log1p(torch.norm(grad)))
        else:
            align_scores[i] = safe_cosine(ref, grad) * float(torch.log1p(torch.norm(grad)))
    for i in range(J):
        for j in range(i + 1, J):
            c = 1.0 - safe_cosine(clean_gradients[i], clean_gradients[j])
            conflict_matrix[i, j] = c
            conflict_matrix[j, i] = c
    pi = np.full(J, 1.0 / J, dtype=np.float32)
    step_size = max(0.05, 0.35 * float(temperature))
    lam_ent = max(0.0, float(entropy_coef))
    for _ in range(max(1, int(steps))):
        conf_grad = conflict_matrix @ pi
        ent_grad = -(np.log(np.clip(pi, 1e-8, 1.0)) + 1.0)
        grad = np.nan_to_num(align_scores - float(beta) * conf_grad + lam_ent * ent_grad, nan=0.0, posinf=0.0, neginf=0.0)
        pi = _project_to_simplex(pi + step_size * grad)
    mixed = sum(float(pi[j]) * clean_gradients[j] for j in range(J))
    mixed = torch.nan_to_num(mixed.detach().clone(), nan=0.0, posinf=0.0, neginf=0.0)
    conflict = 0.0
    for i in range(J):
        for j in range(i + 1, J):
            conflict += float(pi[i] * pi[j]) * float(conflict_matrix[i, j])
    entropy = -float(np.sum(pi * np.log(np.clip(pi, 1e-8, 1.0))))
    objective = float(np.dot(align_scores, pi) - float(beta) * conflict + lam_ent * entropy)
    return pi.astype(np.float32), mixed, float(conflict), objective
