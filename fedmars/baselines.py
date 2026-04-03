from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import FedMARSConfig
from .data import ClientDataset, infer_client_weights
from .utils import clone_model, detach_state_dict, evaluate_classifier, load_state_dict_, move_batch_to_device, percentile, set_seed, unpack_batch


def _stable_client_seed(value: int | str) -> int:
    s = str(value)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(s)) % 1000003


def _count_model_bits(model: torch.nn.Module, param_bits: int = 32) -> int:
    return int(sum(int(p.numel()) for p in model.parameters()) * param_bits)


class _BaseFederatedBaseline:
    def __init__(self, model: torch.nn.Module, config: FedMARSConfig, criterion: torch.nn.Module | None = None):
        self.model = model.to(config.device)
        self.config = config
        self.criterion = criterion if criterion is not None else torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.model_bits = _count_model_bits(self.model, config.param_bits)
        set_seed(config.random_state)

    def _sample_clients(self, clients: Sequence[ClientDataset], round_idx: int) -> list[ClientDataset]:
        rng = np.random.default_rng(self.config.random_state + round_idx)
        num_clients = len(clients)
        choose = max(self.config.min_clients_per_round, int(round(self.config.client_fraction * num_clients)))
        choose = min(max(1, choose), num_clients)
        picked = rng.choice(np.arange(num_clients), size=choose, replace=False)
        return [clients[int(i)] for i in picked.tolist()]

    def _make_loader(self, dataset, seed: int) -> DataLoader:
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=self.config.local_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        return evaluate_classifier(self.model, loader, self.criterion, self.config.device)

    def evaluate_clients(self, clients: Sequence[ClientDataset], batch_size: int = 256) -> dict[str, object]:
        per_client: dict[str, float] = {}
        for client in clients:
            loader = DataLoader(client.dataset, batch_size=batch_size, shuffle=False)
            per_client[str(client.client_id)] = float(self.evaluate(loader)["accuracy"])
        vals = list(per_client.values())
        return {
            "per_client_accuracy": per_client,
            "mean_accuracy": float(np.mean(vals)) if vals else 0.0,
            "std_accuracy": float(np.std(vals)) if vals else 0.0,
            "worst_accuracy": float(np.min(vals)) if vals else 0.0,
            "p10_accuracy": percentile(vals, 10.0),
        }


class FedAvg(_BaseFederatedBaseline):
    def fit(
        self,
        clients: Sequence[ClientDataset],
        server_val_loader: DataLoader | None = None,
        server_test_loader: DataLoader | None = None,
    ) -> dict:
        history = {"config": asdict(self.config), "rounds": []}
        client_weights_all = infer_client_weights(clients)

        for round_idx in range(self.config.num_rounds):
            sampled = self._sample_clients(clients, round_idx)
            global_state = detach_state_dict(self.model)
            client_updates = []
            weights = []

            for client in sampled:
                local_model = clone_model(self.model, self.config.device)
                load_state_dict_(local_model, global_state)
                optimizer = torch.optim.SGD(
                    local_model.parameters(),
                    lr=self.config.rho_max,
                    momentum=0.9,
                    weight_decay=self.config.weight_decay,
                )
                loader = self._make_loader(client.dataset, seed=self.config.random_state + round_idx * 1000 + int(client.client_id if isinstance(client.client_id, int) else abs(hash(str(client.client_id))) % 100000))
                local_model.train()

                for _ in range(self.config.local_epochs):
                    for batch in loader:
                        batch = move_batch_to_device(batch, self.config.device)
                        x, y = unpack_batch(batch)
                        optimizer.zero_grad(set_to_none=True)
                        loss = self.criterion(local_model(x), y)
                        loss.backward()
                        if self.config.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=self.config.max_grad_norm)
                        optimizer.step()

                update = {k: local_model.state_dict()[k].detach().clone().cpu() - global_state[k] for k in global_state}
                client_updates.append(update)
                weights.append(client_weights_all[client.client_id])

            total_weight = float(sum(weights)) or 1.0
            with torch.no_grad():
                named_params = dict(self.model.named_parameters())
                for pname in named_params:
                    delta = torch.zeros_like(named_params[pname])
                    for update, weight in zip(client_updates, weights):
                        delta += (float(weight) / total_weight) * update[pname].to(self.config.device)
                    named_params[pname].add_(delta)

            log = {
                "round": round_idx,
                "communication_ratio": 1.0,
                "client_to_server_bits": int(self.model_bits * len(sampled)),
                "server_to_client_bits": int(self.model_bits * len(sampled)) if self.config.track_server_to_client_bits else 0,
            }
            if server_val_loader is not None:
                log["validation"] = self.evaluate(server_val_loader)
            history["rounds"].append(log)

        if server_test_loader is not None:
            history["test"] = self.evaluate(server_test_loader)
        return history


class FedProx(_BaseFederatedBaseline):
    def fit(
        self,
        clients: Sequence[ClientDataset],
        server_val_loader: DataLoader | None = None,
        server_test_loader: DataLoader | None = None,
        mu: float = 0.01,
    ) -> dict:
        history = {"config": asdict(self.config), "rounds": []}
        client_weights_all = infer_client_weights(clients)

        for round_idx in range(self.config.num_rounds):
            sampled = self._sample_clients(clients, round_idx)
            global_state = detach_state_dict(self.model)
            client_updates = []
            weights = []

            for client in sampled:
                local_model = clone_model(self.model, self.config.device)
                load_state_dict_(local_model, global_state)
                optimizer = torch.optim.SGD(
                    local_model.parameters(),
                    lr=self.config.rho_max,
                    momentum=0.9,
                    weight_decay=self.config.weight_decay,
                )
                loader = self._make_loader(client.dataset, seed=self.config.random_state + round_idx * 1000 + int(client.client_id if isinstance(client.client_id, int) else abs(hash(str(client.client_id))) % 100000))
                local_model.train()
                global_ref = {k: v.detach().clone().to(self.config.device) for k, v in global_state.items()}

                for _ in range(self.config.local_epochs):
                    for batch in loader:
                        batch = move_batch_to_device(batch, self.config.device)
                        x, y = unpack_batch(batch)
                        optimizer.zero_grad(set_to_none=True)
                        loss = self.criterion(local_model(x), y)
                        prox = 0.0
                        named_params = dict(local_model.named_parameters())
                        for pname, param in named_params.items():
                            prox = prox + 0.5 * float(mu) * torch.sum((param - global_ref[pname]) ** 2)
                        total_loss = loss + prox
                        total_loss.backward()
                        if self.config.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=self.config.max_grad_norm)
                        optimizer.step()

                update = {k: local_model.state_dict()[k].detach().clone().cpu() - global_state[k] for k in global_state}
                client_updates.append(update)
                weights.append(client_weights_all[client.client_id])

            total_weight = float(sum(weights)) or 1.0
            with torch.no_grad():
                named_params = dict(self.model.named_parameters())
                for pname in named_params:
                    delta = torch.zeros_like(named_params[pname])
                    for update, weight in zip(client_updates, weights):
                        delta += (float(weight) / total_weight) * update[pname].to(self.config.device)
                    named_params[pname].add_(delta)

            log = {
                "round": round_idx,
                "communication_ratio": 1.0,
                "client_to_server_bits": int(self.model_bits * len(sampled)),
                "server_to_client_bits": int(self.model_bits * len(sampled)) if self.config.track_server_to_client_bits else 0,
            }
            if server_val_loader is not None:
                log["validation"] = self.evaluate(server_val_loader)
            history["rounds"].append(log)

        if server_test_loader is not None:
            history["test"] = self.evaluate(server_test_loader)
        return history
