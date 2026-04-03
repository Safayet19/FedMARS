from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset


@dataclass(slots=True)
class ClientDataset:
    client_id: int | str
    dataset: Dataset
    weight: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.dataset)


@dataclass(slots=True)
class IndexedSubset(Dataset):
    dataset: Dataset
    indices: Sequence[int]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


def make_tensor_client_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    client_id: int | str,
    weight: Optional[float] = None,
) -> ClientDataset:
    return ClientDataset(client_id=client_id, dataset=TensorDataset(x, y), weight=weight)


def extract_targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(getattr(dataset, "targets"))
    if hasattr(dataset, "labels"):
        return np.asarray(getattr(dataset, "labels"))
    ys: list[int] = []
    for i in range(len(dataset)):
        item = dataset[i]
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError("Dataset items must expose labels in position 1.")
        ys.append(int(item[1]))
    return np.asarray(ys)


def infer_client_weights(clients: Sequence[ClientDataset]) -> dict[int | str, float]:
    lengths = np.asarray([len(c) for c in clients], dtype=float)
    total = float(lengths.sum())
    out: dict[int | str, float] = {}
    if total <= 0.0:
        for client in clients:
            out[client.client_id] = 0.0
        return out
    for client, length in zip(clients, lengths):
        out[client.client_id] = float(client.weight) if client.weight is not None else float(length / total)
    return out


def dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_size: int = 8,
) -> list[ClientDataset]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    rng = np.random.default_rng(seed)
    y = extract_targets(dataset)
    classes = np.unique(y)
    client_indices = [[] for _ in range(num_clients)]

    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        props = rng.dirichlet(np.full(num_clients, alpha))
        cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
        shards = np.split(idx, cuts)
        for cid, shard in enumerate(shards):
            client_indices[cid].extend(int(i) for i in shard.tolist())

    for _ in range(100):
        sizes = [len(v) for v in client_indices]
        if min(sizes) >= min_size:
            break
        donor = int(np.argmax(sizes))
        receiver = int(np.argmin(sizes))
        if sizes[donor] <= min_size:
            break
        moved = client_indices[donor].pop()
        client_indices[receiver].append(moved)

    clients: list[ClientDataset] = []
    for cid, indices in enumerate(client_indices):
        clients.append(ClientDataset(client_id=cid, dataset=Subset(dataset, sorted(indices))))
    return clients
