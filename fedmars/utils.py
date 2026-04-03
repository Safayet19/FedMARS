from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def safe_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = torch.norm(a) * torch.norm(b)
    if float(denom) <= eps:
        return 0.0
    return float(torch.dot(a, b) / (denom + eps))


def detach_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}


def load_state_dict_(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    model.load_state_dict({k: v.detach().clone() for k, v in state_dict.items()})


def clone_model(model: torch.nn.Module, device: str | torch.device) -> torch.nn.Module:
    return copy.deepcopy(model).to(device)


def unpack_batch(batch: Any):
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("Each batch must be a tuple or list with at least (x, y).")


def move_batch_to_device(batch: Any, device: str | torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    return batch


def evaluate_classifier(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: str | torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            x, y = unpack_batch(batch)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss) * len(y)
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == y).sum())
            total_examples += int(len(y))
    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))
