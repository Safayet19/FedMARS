from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import FedMARSConfig
from .data import ClientDataset, infer_client_weights
from .utils import clone_model, detach_state_dict, evaluate_classifier, load_state_dict_, move_batch_to_device, set_seed, unpack_batch


class _BaseFederatedBaseline:
    def __init__(self, model: torch.nn.Module, config: FedMARSConfig, criterion: torch.nn.Module | None = None):
        self.model = model.to(config.device)
        self.config = config
        self.criterion = criterion if criterion is not None else torch.nn.CrossEntropyLoss()
        set_seed(config.random_state)

    def _sample_clients(self, clients: Sequence[ClientDataset], round_idx: int) -> list[ClientDataset]:
        rng = np.random.default_rng(self.config.random_state + round_idx)
        num_clients = len(clients)
        choose = max(self.config.min_clients_per_round, int(round(self.config.client_fraction * num_clients)))
        choose = min(max(1, choose), num_clients)
        picked = rng.choice(np.arange(num_clients), size=choose, replace=False)
        return [clients[int(i)] for i in picked.tolist()]

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        return evaluate_classifier(self.model, loader, self.criterion, self.config.device)


class FedAvg(_BaseFederatedBaseline):
    def fit(self, clients: Sequence[ClientDataset], server_val_loader: DataLoader | None = None) -> dict:
        history = {"rounds": []}
        client_weights_all = infer_client_weights(clients)
        for round_idx in range(self.config.num_rounds):
            sampled = self._sample_clients(clients, round_idx)
            global_state = detach_state_dict(self.model)
            client_updates = []
            weights = []
            for client in sampled:
                local_model = clone_model(self.model, self.config.device)
                load_state_dict_(local_model, global_state)
                optimizer = torch.optim.SGD(local_model.parameters(), lr=self.config.rho_max)
                loader = DataLoader(client.dataset, batch_size=self.config.local_batch_size, shuffle=True)
                local_model.train()
                for _ in range(self.config.local_epochs):
                    for batch in loader:
                        batch = move_batch_to_device(batch, self.config.device)
                        x, y = unpack_batch(batch)
                        optimizer.zero_grad(set_to_none=True)
                        loss = self.criterion(local_model(x), y)
                        loss.backward()
                        optimizer.step()
                update = {k: local_model.state_dict()[k].detach().clone() - global_state[k] for k in global_state}
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
            log = {"round": round_idx}
            if server_val_loader is not None:
                log.update(self.evaluate(server_val_loader))
            history["rounds"].append(log)
        return history


class FedProx(_BaseFederatedBaseline):
    def fit(self, clients: Sequence[ClientDataset], server_val_loader: DataLoader | None = None, mu: float = 0.01) -> dict:
        history = {"rounds": []}
        client_weights_all = infer_client_weights(clients)
        for round_idx in range(self.config.num_rounds):
            sampled = self._sample_clients(clients, round_idx)
            global_state = detach_state_dict(self.model)
            client_updates = []
            weights = []
            for client in sampled:
                local_model = clone_model(self.model, self.config.device)
                load_state_dict_(local_model, global_state)
                optimizer = torch.optim.SGD(local_model.parameters(), lr=self.config.rho_max)
                loader = DataLoader(client.dataset, batch_size=self.config.local_batch_size, shuffle=True)
                local_model.train()
                named_params = dict(local_model.named_parameters())
                for _ in range(self.config.local_epochs):
                    for batch in loader:
                        batch = move_batch_to_device(batch, self.config.device)
                        x, y = unpack_batch(batch)
                        optimizer.zero_grad(set_to_none=True)
                        logits = local_model(x)
                        loss = self.criterion(logits, y)
                        prox = 0.0
                        for pname, param in named_params.items():
                            prox = prox + 0.5 * mu * torch.sum((param - global_state[pname].to(param.device)) ** 2)
                        (loss + prox).backward()
                        optimizer.step()
                update = {k: local_model.state_dict()[k].detach().clone() - global_state[k] for k in global_state}
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
            log = {"round": round_idx}
            if server_val_loader is not None:
                log.update(self.evaluate(server_val_loader))
            history["rounds"].append(log)
        return history
