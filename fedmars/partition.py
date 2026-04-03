from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data._utils.collate import default_collate

from .data import extract_targets
from .utils import unpack_batch


def _tensorize_inputs(dataset, indices: Sequence[int]) -> np.ndarray:
    rows = []
    for idx in indices:
        x, _ = unpack_batch(dataset[idx])
        rows.append(torch.as_tensor(x).reshape(-1).float().cpu().numpy())
    return np.stack(rows, axis=0)


def _round_robin_merge(groups: list[list[int]], target_groups: int) -> list[list[int]]:
    groups = [g for g in groups if g]
    if not groups:
        return []
    if len(groups) <= target_groups:
        return groups
    groups = sorted(groups, key=len, reverse=True)
    merged = [[] for _ in range(target_groups)]
    for idx, group in enumerate(groups):
        merged[idx % target_groups].extend(group)
    return [sorted(g) for g in merged if g]


def build_local_modes(
    dataset,
    num_clusters: int,
    method: str,
    seed: int,
    max_samples: int,
    min_examples_for_multimodal: int,
) -> list[list[int]]:
    n = len(dataset)
    if num_clusters <= 1 or n < max(min_examples_for_multimodal, num_clusters * 2):
        return [list(range(n))]
    if method == "single":
        return [list(range(n))]
    if method == "label":
        y = extract_targets(dataset)
        groups = [np.where(y == cls)[0].tolist() for cls in np.unique(y)]
        groups = _round_robin_merge(groups, num_clusters)
        return [sorted(g) for g in groups if g]
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n)
    if method == "random":
        rng.shuffle(all_indices)
        return [sorted(g.tolist()) for g in np.array_split(all_indices, num_clusters) if len(g) > 0]
    if method == "kmeans":
        sample_idx = all_indices
        if n > max_samples:
            sample_idx = np.sort(rng.choice(all_indices, size=max_samples, replace=False))
        X_sample = _tensorize_inputs(dataset, sample_idx)
        k = min(num_clusters, len(sample_idx))
        if k <= 1:
            return [list(range(n))]
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(X_sample)
        X_all = _tensorize_inputs(dataset, all_indices)
        labels = km.predict(X_all)
        groups = [all_indices[labels == i].tolist() for i in range(k)]
        groups = [sorted(g) for g in groups if g]
        return groups or [list(range(n))]
    raise ValueError(f"Unsupported partition_method: {method}")


def sample_batch_from_indices(dataset, indices: Sequence[int], batch_size: int, seed: int):
    if len(indices) == 0:
        raise ValueError("Cannot sample from an empty index set.")
    rng = np.random.default_rng(seed)
    replace = len(indices) < batch_size
    chosen = rng.choice(np.asarray(indices), size=batch_size, replace=replace)
    batch = [dataset[int(i)] for i in chosen.tolist()]
    return default_collate(batch)


def sample_probe_batches(dataset, batch_size: int, seed: int):
    all_indices = list(range(len(dataset)))
    return (
        sample_batch_from_indices(dataset, all_indices, batch_size, seed=seed),
        sample_batch_from_indices(dataset, all_indices, batch_size, seed=seed + 1),
    )