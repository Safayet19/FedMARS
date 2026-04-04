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


def _zero_state_like(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v).detach().clone().cpu() for k, v in model.state_dict().items()}


def _state_update(local_state: dict[str, torch.Tensor], global_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: local_state[k].detach().clone().cpu() - global_state[k].detach().clone().cpu() for k in global_state}


def _weighted_average_updates(updates: list[dict[str, torch.Tensor]], weights: list[float], ref_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    if total <= 0.0:
        return {k: torch.zeros_like(v) for k, v in ref_state.items()}
    out = {k: torch.zeros_like(v) for k, v in ref_state.items()}
    for update, weight in zip(updates, weights):
        for k in out:
            out[k] += float(weight) / total * update[k]
    return out


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
        return DataLoader(dataset, batch_size=self.config.local_batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=self.config.num_workers, pin_memory=self.config.pin_memory)

    def _make_optimizer(self, model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=self.config.weight_decay)

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        return evaluate_classifier(self.model, loader, self.criterion, self.config.device)

    def evaluate_clients(self, clients: Sequence[ClientDataset], batch_size: int = 256) -> dict[str, object]:
        per_client = {}
        for client in clients:
            loader = DataLoader(client.dataset, batch_size=batch_size, shuffle=False)
            per_client[str(client.client_id)] = float(self.evaluate(loader)["accuracy"])
        vals = list(per_client.values())
        return {"per_client_accuracy": per_client, "mean_accuracy": float(np.mean(vals)) if vals else 0.0, "std_accuracy": float(np.std(vals)) if vals else 0.0, "worst_accuracy": float(np.min(vals)) if vals else 0.0, "p10_accuracy": percentile(vals, 10.0)}

    def _local_sgd_state(self, client: ClientDataset, global_state: dict[str, torch.Tensor], round_idx: int, lr: float) -> tuple[dict[str, torch.Tensor], float]:
        local_model = clone_model(self.model, self.config.device)
        load_state_dict_(local_model, global_state)
        optimizer = self._make_optimizer(local_model, lr)
        loader = self._make_loader(client.dataset, self.config.random_state + round_idx * 1000 + _stable_client_seed(client.client_id))
        local_model.train()
        total_loss = 0.0
        total_examples = 0
        for _ in range(self.config.local_epochs):
            for batch in loader:
                x, y = unpack_batch(move_batch_to_device(batch, self.config.device))
                optimizer.zero_grad(set_to_none=True)
                loss = self.criterion(local_model(x), y)
                loss.backward()
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=self.config.max_grad_norm)
                optimizer.step()
                total_loss += float(loss.detach()) * len(y)
                total_examples += int(len(y))
        return detach_state_dict(local_model), float(total_loss / max(total_examples, 1))

    def _prox_local_state(self, client: ClientDataset, global_state: dict[str, torch.Tensor], round_idx: int, lr: float, mu: float) -> tuple[dict[str, torch.Tensor], float]:
        local_model = clone_model(self.model, self.config.device)
        load_state_dict_(local_model, global_state)
        optimizer = self._make_optimizer(local_model, lr)
        loader = self._make_loader(client.dataset, self.config.random_state + round_idx * 1000 + _stable_client_seed(client.client_id))
        global_ref = {k: v.detach().clone().to(self.config.device) for k, v in global_state.items()}
        local_model.train()
        total_loss = 0.0
        total_examples = 0
        for _ in range(self.config.local_epochs):
            for batch in loader:
                x, y = unpack_batch(move_batch_to_device(batch, self.config.device))
                optimizer.zero_grad(set_to_none=True)
                loss = self.criterion(local_model(x), y)
                prox = 0.0
                for pname, param in local_model.named_parameters():
                    prox = prox + 0.5 * float(mu) * torch.sum((param - global_ref[pname]) ** 2)
                total = loss + prox
                total.backward()
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=self.config.max_grad_norm)
                optimizer.step()
                total_loss += float(loss.detach()) * len(y)
                total_examples += int(len(y))
        return detach_state_dict(local_model), float(total_loss / max(total_examples, 1))

    def _round_log(self, round_idx: int, sampled_clients: Sequence[ClientDataset], validation: dict[str, float] | None = None) -> dict:
        log = {"round": round_idx, "sampled_clients": [client.client_id for client in sampled_clients], "communication_ratio": 1.0, "client_to_server_bits": int(self.model_bits * len(sampled_clients)), "server_to_client_bits": int(self.model_bits * len(sampled_clients)) if self.config.track_server_to_client_bits else 0}
        log["total_bits"] = int(log["client_to_server_bits"] + log["server_to_client_bits"])
        if validation is not None:
            log["validation"] = validation
        return log


class FedAvg(_BaseFederatedBaseline):
    def fit(self, clients: Sequence[ClientDataset], server_val_loader: DataLoader | None = None, server_test_loader: DataLoader | None = None) -> dict:
        history = {"config": asdict(self.config), "rounds": []}
        client_weights_all = infer_client_weights(clients)
        lr = float(self.config.rho_max)
        for round_idx in range(self.config.num_rounds):
            sampled = self._sample_clients(clients, round_idx)
            global_state = detach_state_dict(self.model)
            updates = []
            weights = []
            for client in sampled:
                local_state, _ = self._local_sgd_state(client, global_state, round_idx, lr)
                updates.append(_state_update(local_state, global_state))
                weights.append(client_weights_all[client.client_id])
            avg_update = _weighted_average_updates(updates, weights, global_state)
            with torch.no_grad():
                for pname, param in self.model.named_parameters():
                    param.add_(avg_update[pname].to(self.config.device))
            validation = self.evaluate(server_val_loader) if server_val_loader is not None else None
            history["rounds"].append(self._round_log(round_idx, sampled, validation))
        if server_test_loader is not None:
            history["test"] = self.evaluate(server_test_loader)
        return history


class FedProx(_BaseFederatedBaseline):
    def fit(self, clients: Sequence[ClientDataset], server_val_loader: DataLoader | None = None, server_test_loader: DataLoader | None = None, mu: float = 0.01) -> dict:
        history = {"config": asdict(self.config), "rounds": []}
        client_weights_all = infer_client_weights(clients)
        lr = float(self.config.rho_max)
        for round_idx in range(self.config.num_rounds):
            sampled = self._sample_clients(clients, round_idx)
            global_state = detach_state_dict(self.model)
            updates = []
            weights = []
            for client in sampled:
                local_state, _ = self._prox_local_state(client, global_state, round_idx, lr, mu)
                updates.append(_state_update(local_state, global_state))
                weights.append(client_weights_all[client.client_id])
            avg_update = _weighted_average_updates(updates, weights, global_state)
            with torch.no_grad():
                for pname, param in self.model.named_parameters():
                    param.add_(avg_update[pname].to(self.config.device))
            validation = self.evaluate(server_val_loader) if server_val_loader is not None else None
            history["rounds"].append(self._round_log(round_idx, sampled, validation))
        if server_test_loader is not None:
            history["test"] = self.evaluate(server_test_loader)
        return history


class FedOpt(_BaseFederatedBaseline):
    def fit(self, clients: Sequence[ClientDataset], server_val_loader: DataLoader | None = None, server_test_loader: DataLoader | None = None, server_lr: float = 1.0, beta1: float = 0.9, beta2: float = 0.99, tau: float = 1e-3) -> dict:
        history = {"config": asdict(self.config), "rounds": []}
        client_weights_all = infer_client_weights(clients)
        lr = float(self.config.rho_max)
        m = _zero_state_like(self.model)
        v = _zero_state_like(self.model)
        for round_idx in range(self.config.num_rounds):
            sampled = self._sample_clients(clients, round_idx)
            global_state = detach_state_dict(self.model)
            updates = []
            weights = []
            for client in sampled:
                local_state, _ = self._local_sgd_state(client, global_state, round_idx, lr)
                updates.append(_state_update(local_state, global_state))
                weights.append(client_weights_all[client.client_id])
            avg_update = _weighted_average_updates(updates, weights, global_state)
            with torch.no_grad():
                for pname, param in self.model.named_parameters():
                    delta = avg_update[pname].to(self.config.device)
                    m[pname] = beta1 * m[pname] + (1.0 - beta1) * delta.detach().cpu()
                    v[pname] = beta2 * v[pname] + (1.0 - beta2) * (delta.detach().cpu() ** 2)
                    m_hat = m[pname].to(self.config.device) / (1.0 - beta1 ** (round_idx + 1))
                    v_hat = v[pname].to(self.config.device) / (1.0 - beta2 ** (round_idx + 1))
                    param.add_(float(server_lr) * m_hat / (torch.sqrt(v_hat) + float(tau)))
            validation = self.evaluate(server_val_loader) if server_val_loader is not None else None
            history["rounds"].append(self._round_log(round_idx, sampled, validation))
        if server_test_loader is not None:
            history["test"] = self.evaluate(server_test_loader)
        return history


class SCAFFOLD(_BaseFederatedBaseline):
    def fit(self, clients: Sequence[ClientDataset], server_val_loader: DataLoader | None = None, server_test_loader: DataLoader | None = None, lr: float | None = None) -> dict:
        history = {"config": asdict(self.config), "rounds": []}
        lr = float(self.config.rho_max if lr is None else lr)
        client_weights_all = infer_client_weights(clients)
        server_c = _zero_state_like(self.model)
        client_c = {client.client_id: _zero_state_like(self.model) for client in clients}
        for round_idx in range(self.config.num_rounds):
            sampled = self._sample_clients(clients, round_idx)
            global_state = detach_state_dict(self.model)
            updates = []
            weights = []
            delta_cs = []
            for client in sampled:
                local_model = clone_model(self.model, self.config.device)
                load_state_dict_(local_model, global_state)
                loader = self._make_loader(client.dataset, self.config.random_state + round_idx * 1000 + _stable_client_seed(client.client_id))
                steps = 0
                local_model.train()
                named_params = dict(local_model.named_parameters())
                for _ in range(self.config.local_epochs):
                    for batch in loader:
                        x, y = unpack_batch(move_batch_to_device(batch, self.config.device))
                        local_model.zero_grad(set_to_none=True)
                        loss = self.criterion(local_model(x), y)
                        loss.backward()
                        if self.config.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=self.config.max_grad_norm)
                        with torch.no_grad():
                            for pname, param in named_params.items():
                                grad = torch.zeros_like(param) if param.grad is None else param.grad
                                control = client_c[client.client_id][pname].to(self.config.device) - server_c[pname].to(self.config.device)
                                param.add_(-lr * (grad - control + self.config.weight_decay * param))
                        steps += 1
                steps = max(steps, 1)
                local_state = detach_state_dict(local_model)
                updates.append(_state_update(local_state, global_state))
                weights.append(client_weights_all[client.client_id])
                c_new = {}
                delta_c = {}
                for pname in global_state:
                    c_new[pname] = client_c[client.client_id][pname] - server_c[pname] + (global_state[pname] - local_state[pname]) / float(steps * lr)
                    delta_c[pname] = c_new[pname] - client_c[client.client_id][pname]
                client_c[client.client_id] = c_new
                delta_cs.append(delta_c)
            avg_update = _weighted_average_updates(updates, weights, global_state)
            with torch.no_grad():
                for pname, param in self.model.named_parameters():
                    param.add_(avg_update[pname].to(self.config.device))
            sampled_fraction = float(len(sampled) / max(len(clients), 1))
            avg_delta_c = _weighted_average_updates(delta_cs, [1.0 for _ in delta_cs], global_state)
            for pname in server_c:
                server_c[pname] = server_c[pname] + sampled_fraction * avg_delta_c[pname]
            validation = self.evaluate(server_val_loader) if server_val_loader is not None else None
            history["rounds"].append(self._round_log(round_idx, sampled, validation))
        if server_test_loader is not None:
            history["test"] = self.evaluate(server_test_loader)
        return history


class FedDyn(_BaseFederatedBaseline):
    def fit(self, clients: Sequence[ClientDataset], server_val_loader: DataLoader | None = None, server_test_loader: DataLoader | None = None, alpha: float = 0.01, lr: float | None = None) -> dict:
        history = {"config": asdict(self.config), "rounds": []}
        lr = float(self.config.rho_max if lr is None else lr)
        client_weights_all = infer_client_weights(clients)
        dual = {client.client_id: _zero_state_like(self.model) for client in clients}
        for round_idx in range(self.config.num_rounds):
            sampled = self._sample_clients(clients, round_idx)
            global_state = detach_state_dict(self.model)
            updates = []
            weights = []
            for client in sampled:
                local_model = clone_model(self.model, self.config.device)
                load_state_dict_(local_model, global_state)
                optimizer = self._make_optimizer(local_model, lr)
                loader = self._make_loader(client.dataset, self.config.random_state + round_idx * 1000 + _stable_client_seed(client.client_id))
                global_ref = {k: v.detach().clone().to(self.config.device) for k, v in global_state.items()}
                local_model.train()
                for _ in range(self.config.local_epochs):
                    for batch in loader:
                        x, y = unpack_batch(move_batch_to_device(batch, self.config.device))
                        optimizer.zero_grad(set_to_none=True)
                        loss = self.criterion(local_model(x), y)
                        reg = 0.0
                        lin = 0.0
                        for pname, param in local_model.named_parameters():
                            reg = reg + 0.5 * float(alpha) * torch.sum((param - global_ref[pname]) ** 2)
                            lin = lin + torch.sum(dual[client.client_id][pname].to(self.config.device) * param)
                        total = loss + reg - lin
                        total.backward()
                        if self.config.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=self.config.max_grad_norm)
                        optimizer.step()
                local_state = detach_state_dict(local_model)
                for pname in dual[client.client_id]:
                    dual[client.client_id][pname] = dual[client.client_id][pname] - float(alpha) * (local_state[pname] - global_state[pname])
                updates.append(_state_update(local_state, global_state))
                weights.append(client_weights_all[client.client_id])
            avg_update = _weighted_average_updates(updates, weights, global_state)
            with torch.no_grad():
                for pname, param in self.model.named_parameters():
                    param.add_(avg_update[pname].to(self.config.device))
            validation = self.evaluate(server_val_loader) if server_val_loader is not None else None
            history["rounds"].append(self._round_log(round_idx, sampled, validation))
        if server_test_loader is not None:
            history["test"] = self.evaluate(server_test_loader)
        return history


class QFFL(_BaseFederatedBaseline):
    def fit(self, clients: Sequence[ClientDataset], server_val_loader: DataLoader | None = None, server_test_loader: DataLoader | None = None, q: float = 0.5, lr: float | None = None) -> dict:
        history = {"config": asdict(self.config), "rounds": []}
        lr = float(self.config.rho_max if lr is None else lr)
        client_weights_all = infer_client_weights(clients)
        for round_idx in range(self.config.num_rounds):
            sampled = self._sample_clients(clients, round_idx)
            global_state = detach_state_dict(self.model)
            updates = []
            weights = []
            for client in sampled:
                local_state, local_loss = self._local_sgd_state(client, global_state, round_idx, lr)
                updates.append(_state_update(local_state, global_state))
                weights.append(client_weights_all[client.client_id] * float(max(local_loss, 1e-8) ** q))
            avg_update = _weighted_average_updates(updates, weights, global_state)
            with torch.no_grad():
                for pname, param in self.model.named_parameters():
                    param.add_(avg_update[pname].to(self.config.device))
            validation = self.evaluate(server_val_loader) if server_val_loader is not None else None
            history["rounds"].append(self._round_log(round_idx, sampled, validation))
        if server_test_loader is not None:
            history["test"] = self.evaluate(server_test_loader)
        return history


class Ditto(_BaseFederatedBaseline):
    def __init__(self, model: torch.nn.Module, config: FedMARSConfig, criterion: torch.nn.Module | None = None):
        super().__init__(model, config, criterion)
        self.personalized_models: dict[int | str, dict[str, torch.Tensor]] = {}

    def fit(self, clients: Sequence[ClientDataset], server_val_loader: DataLoader | None = None, server_test_loader: DataLoader | None = None, mu: float = 0.01, lr: float | None = None) -> dict:
        history = {"config": asdict(self.config), "rounds": []}
        lr = float(self.config.rho_max if lr is None else lr)
        client_weights_all = infer_client_weights(clients)
        if not self.personalized_models:
            state = detach_state_dict(self.model)
            self.personalized_models = {client.client_id: {k: v.clone() for k, v in state.items()} for client in clients}
        for round_idx in range(self.config.num_rounds):
            sampled = self._sample_clients(clients, round_idx)
            global_state = detach_state_dict(self.model)
            updates = []
            weights = []
            for client in sampled:
                local_state, _ = self._local_sgd_state(client, global_state, round_idx, lr)
                updates.append(_state_update(local_state, global_state))
                weights.append(client_weights_all[client.client_id])
            avg_update = _weighted_average_updates(updates, weights, global_state)
            with torch.no_grad():
                for pname, param in self.model.named_parameters():
                    param.add_(avg_update[pname].to(self.config.device))
            new_global_state = detach_state_dict(self.model)
            for client in sampled:
                personal_model = clone_model(self.model, self.config.device)
                load_state_dict_(personal_model, self.personalized_models[client.client_id])
                optimizer = self._make_optimizer(personal_model, lr)
                loader = self._make_loader(client.dataset, self.config.random_state + 5000 + round_idx * 1000 + _stable_client_seed(client.client_id))
                global_ref = {k: v.detach().clone().to(self.config.device) for k, v in new_global_state.items()}
                personal_model.train()
                for _ in range(self.config.local_epochs):
                    for batch in loader:
                        x, y = unpack_batch(move_batch_to_device(batch, self.config.device))
                        optimizer.zero_grad(set_to_none=True)
                        loss = self.criterion(personal_model(x), y)
                        prox = 0.0
                        for pname, param in personal_model.named_parameters():
                            prox = prox + 0.5 * float(mu) * torch.sum((param - global_ref[pname]) ** 2)
                        total = loss + prox
                        total.backward()
                        if self.config.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(personal_model.parameters(), max_norm=self.config.max_grad_norm)
                        optimizer.step()
                self.personalized_models[client.client_id] = detach_state_dict(personal_model)
            validation = self.evaluate(server_val_loader) if server_val_loader is not None else None
            history["rounds"].append(self._round_log(round_idx, sampled, validation))
        if server_test_loader is not None:
            history["test"] = self.evaluate(server_test_loader)
        return history

    def evaluate_personalized(self, clients: Sequence[ClientDataset], batch_size: int = 256) -> dict[str, object]:
        per_client = {}
        for client in clients:
            model = clone_model(self.model, self.config.device)
            state = self.personalized_models.get(client.client_id)
            if state is not None:
                load_state_dict_(model, state)
            loader = DataLoader(client.dataset, batch_size=batch_size, shuffle=False)
            per_client[str(client.client_id)] = float(evaluate_classifier(model, loader, self.criterion, self.config.device)["accuracy"])
        vals = list(per_client.values())
        return {"per_client_accuracy": per_client, "mean_accuracy": float(np.mean(vals)) if vals else 0.0, "std_accuracy": float(np.std(vals)) if vals else 0.0, "worst_accuracy": float(np.min(vals)) if vals else 0.0, "p10_accuracy": percentile(vals, 10.0)}
