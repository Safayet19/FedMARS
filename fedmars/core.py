from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .aggregation import aggregate_sparse_updates, apply_global_update, select_layers_under_budget
from .config import FedMARSConfig
from .controller import AdaptiveRoundController, ControllerAction, RoundState
from .credit import aggregate_global_credit, build_reference_sketch, compute_layer_credit
from .data import ClientDataset, infer_client_weights
from .layers import LayerSpec, build_layer_specs, compute_depth_weights, compute_layer_bits, compute_layer_costs, flatten_grads_from_model, flatten_params_from_state, layer_name_to_spec, state_delta_by_layer
from .mixture import select_counterfactual_mixture
from .partition import build_local_modes, sample_batch_from_indices
from .utils import clone_model, detach_state_dict, evaluate_classifier, load_state_dict_, mean_or_zero, move_batch_to_device, safe_cosine, set_seed, sigmoid, unpack_batch


def _stable_client_seed(value: int | str) -> int:
    s = str(value)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(s)) % 1000003


def _unflatten_layer_vector(vector: torch.Tensor, spec: LayerSpec, state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    offset = 0
    vec = vector.reshape(-1)
    for pname in spec.param_names:
        ref = state_dict[pname]
        numel = int(ref.numel())
        out[pname] = vec[offset : offset + numel].reshape(ref.shape).to(dtype=ref.dtype)
        offset += numel
    return out


def _normalize_global_credit(raw_credit: Mapping[str, float]) -> dict[str, float]:
    if not raw_credit:
        return {}
    vals = np.asarray(list(raw_credit.values()), dtype=float)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-8
    return {name: float(np.clip((float(value) - med) / mad, -2.5, 2.5)) for name, value in raw_credit.items()}


class FedMARS:
    def __init__(self, model: torch.nn.Module, config: FedMARSConfig | None = None, criterion: torch.nn.Module | None = None):
        self.config = config if config is not None else FedMARSConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.criterion = criterion if criterion is not None else torch.nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        set_seed(self.config.random_state)
        self.layer_specs = build_layer_specs(self.model)
        self.layer_map = layer_name_to_spec(self.layer_specs)
        self.depth_weights = compute_depth_weights(self.layer_specs, mode=self.config.depth_weight_mode)
        self.layer_costs = compute_layer_costs(self.layer_specs)
        self.layer_bits = compute_layer_bits(self.layer_specs, param_bits=self.config.param_bits)
        self.model_bits = int(sum(self.layer_bits.values()))
        self.controller = AdaptiveRoundController(self.config.controller, seed=self.config.random_state)
        named_params = dict(self.model.named_parameters())
        self.ref_memory = {spec.name: None for spec in self.layer_specs}
        self.server_velocity = {
            spec.name: {pname: torch.zeros_like(named_params[pname]).detach().clone().cpu() for pname in spec.param_names}
            for spec in self.layer_specs
        }
        self.history: dict[str, Any] = {"config": asdict(self.config), "rounds": []}

    def _sample_clients(self, clients: Sequence[ClientDataset], round_idx: int) -> list[ClientDataset]:
        rng = np.random.default_rng(self.config.random_state + round_idx)
        num_clients = len(clients)
        choose = max(self.config.min_clients_per_round, int(round(self.config.client_fraction * num_clients)))
        choose = min(max(1, choose), num_clients)
        picked = rng.choice(np.arange(num_clients), size=choose, replace=False)
        return [clients[int(i)] for i in picked.tolist()]

    def _build_reference_sketches(self, current_state: Mapping[str, torch.Tensor], previous_state: Mapping[str, torch.Tensor] | None) -> dict[str, torch.Tensor | None]:
        if previous_state is None:
            refs = {spec.name: None for spec in self.layer_specs}
            self.ref_memory = refs.copy()
            return refs
        refs: dict[str, torch.Tensor | None] = {}
        for spec in self.layer_specs:
            cur = flatten_params_from_state(current_state, spec)
            prev = flatten_params_from_state(previous_state, spec)
            unit = build_reference_sketch(cur - prev, mode=self.config.reference_sketch_mode)
            if unit is None:
                refs[spec.name] = self.ref_memory[spec.name]
                continue
            if self.config.reference_sketch_mode == "ema_unit":
                prev_ref = self.ref_memory[spec.name]
                if prev_ref is None:
                    merged = unit
                else:
                    merged = self.config.reference_momentum * prev_ref + (1.0 - self.config.reference_momentum) * unit
                    merged = merged / (torch.norm(merged) + 1e-12)
                refs[spec.name] = merged
            else:
                refs[spec.name] = unit
        self.ref_memory = {k: (v.detach().clone() if v is not None else None) for k, v in refs.items()}
        return refs

    def _compute_batch_loss(self, model: torch.nn.Module, batch) -> float:
        model.eval()
        with torch.no_grad():
            x, y = unpack_batch(move_batch_to_device(batch, self.device))
            return float(self.criterion(model(x), y).detach())

    def _compute_batch_layer_grads(self, model: torch.nn.Module, batch) -> dict[str, torch.Tensor]:
        model.train()
        model.zero_grad(set_to_none=True)
        x, y = unpack_batch(move_batch_to_device(batch, self.device))
        self.criterion(model(x), y).backward()
        return {spec.name: flatten_grads_from_model(model, spec).detach().cpu() for spec in self.layer_specs}

    def _compute_transfer_scores(self, client: ClientDataset, global_state: Mapping[str, torch.Tensor], round_seed: int) -> dict[str, float]:
        num_batches = max(2, int(self.config.transfer_probe_batches))
        probe_grads = []
        all_indices = list(range(len(client.dataset)))
        for batch_idx in range(num_batches):
            batch = sample_batch_from_indices(client.dataset, all_indices, self.config.probe_batch_size, seed=round_seed + batch_idx)
            probe_model = clone_model(self.model, self.device)
            load_state_dict_(probe_model, global_state)
            probe_grads.append(self._compute_batch_layer_grads(probe_model, batch))
        out: dict[str, float] = {}
        for spec in self.layer_specs:
            vals = []
            for i in range(len(probe_grads)):
                for j in range(i + 1, len(probe_grads)):
                    vals.append(safe_cosine(probe_grads[i][spec.name], probe_grads[j][spec.name]))
            out[spec.name] = mean_or_zero(vals)
        return out

    def _map_transfer_to_lr(self, transfer_score: float) -> float:
        if not self.config.ablations.use_transfer_lr:
            return float(self.config.rho_max)
        return float(self.config.rho_min + (self.config.rho_max - self.config.rho_min) * sigmoid(self.config.kappa_transfer * (transfer_score - self.config.tau_transfer)))

    def _probe_layer_gain(self, global_state: Mapping[str, torch.Tensor], spec: LayerSpec, mixed_gradient: torch.Tensor, probe_batch) -> float:
        if float(self.config.probe_step) <= 0.0:
            return 0.0
        base_model = clone_model(self.model, self.device)
        load_state_dict_(base_model, global_state)
        loss_before = self._compute_batch_loss(base_model, probe_batch)
        step_chunks = _unflatten_layer_vector(mixed_gradient, spec, global_state)
        with torch.no_grad():
            named_params = dict(base_model.named_parameters())
            for pname in spec.param_names:
                named_params[pname].add_(-float(self.config.probe_step) * step_chunks[pname].to(named_params[pname].device))
        loss_after = self._compute_batch_loss(base_model, probe_batch)
        return float(loss_before - loss_after)

    def _phase_a_client_credit(self, client: ClientDataset, global_state: Mapping[str, torch.Tensor], ref_sketches: Mapping[str, torch.Tensor | None], round_idx: int) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        client_seed = _stable_client_seed(client.client_id)
        if self.config.ablations.use_multimodal_partition:
            groups = build_local_modes(client.dataset, self.config.num_clusters, self.config.partition_method, self.config.random_state + round_idx * 1000 + client_seed, self.config.max_partition_samples, self.config.min_examples_for_multimodal)
        else:
            groups = [list(range(len(client.dataset)))]
        cluster_grads = []
        repeats = max(1, int(self.config.num_batches_per_cluster))
        for group_idx, indices in enumerate(groups):
            accum = None
            for batch_idx in range(repeats):
                batch = sample_batch_from_indices(client.dataset, indices, self.config.local_batch_size, seed=self.config.random_state + round_idx * 10000 + group_idx * 101 + batch_idx + client_seed)
                probe_model = clone_model(self.model, self.device)
                load_state_dict_(probe_model, global_state)
                grads = self._compute_batch_layer_grads(probe_model, batch)
                if accum is None:
                    accum = {name: tensor.clone() for name, tensor in grads.items()}
                else:
                    for name, tensor in grads.items():
                        accum[name] += tensor
            if accum is not None:
                cluster_grads.append({name: tensor / float(repeats) for name, tensor in accum.items()})
        if not cluster_grads:
            cluster_grads = [{spec.name: torch.zeros(spec.numel) for spec in self.layer_specs}]
        probe_batch = sample_batch_from_indices(client.dataset, list(range(len(client.dataset))), self.config.probe_batch_size, seed=self.config.random_state + round_idx * 20000 + client_seed)
        credits = {}
        details = {}
        for spec in self.layer_specs:
            grads = [cg[spec.name] for cg in cluster_grads]
            reference = ref_sketches.get(spec.name) if self.config.ablations.use_reference_sketch else None
            if self.config.ablations.use_counterfactual_mixture:
                weights, mixed, conflict, objective = select_counterfactual_mixture(grads, reference, self.config.mixture_conflict_beta, self.config.mixture_temperature, steps=self.config.mixture_steps)
            else:
                weights = np.ones(len(grads), dtype=np.float32) / max(len(grads), 1)
                mixed = sum(grads) / max(len(grads), 1)
                conflict = 0.0
                objective = float(torch.norm(mixed))
            record = compute_layer_credit(
                reference=reference,
                mixed_gradient=mixed,
                residual_conflict=conflict if self.config.ablations.use_layer_credit else 0.0,
                cost=self.layer_costs[spec.name] if self.config.ablations.use_layer_credit else 0.0,
                depth_weight=self.depth_weights[spec.name] if self.config.ablations.use_depth_weight else 1.0,
                lambda_r=self.config.lambda_r if self.config.ablations.use_layer_credit else 0.0,
                lambda_c=self.config.lambda_c if self.config.ablations.use_layer_credit else 0.0,
                probe_gain=self._probe_layer_gain(global_state, spec, mixed, probe_batch) if self.config.lambda_v > 0.0 else 0.0,
                lambda_v=self.config.lambda_v if self.config.ablations.use_layer_credit else 0.0,
            )
            credits[spec.name] = record.credit if self.config.ablations.use_layer_credit else record.benefit
            details[spec.name] = {
                "weights": [float(x) for x in weights.tolist()],
                "conflict": float(conflict),
                "objective": float(objective),
                "credit": float(credits[spec.name]),
                "benefit": float(record.benefit),
                "risk": float(record.risk),
                "alignment": float(record.alignment),
                "norm": float(record.norm),
                "probe_gain": float(record.probe_gain),
            }
        return credits, details

    def _phase_b_client_update(self, client: ClientDataset, global_state: Mapping[str, torch.Tensor], selected_layers: Sequence[str], proximal_strengths: Mapping[str, float], round_idx: int) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float], dict[str, float]]:
        local_model = clone_model(self.model, self.device)
        load_state_dict_(local_model, global_state)
        named_params = dict(local_model.named_parameters())
        transfer_scores = self._compute_transfer_scores(client, global_state, round_seed=self.config.random_state + 30000 + round_idx + _stable_client_seed(client.client_id))
        layer_lrs = {name: self._map_transfer_to_lr(score) for name, score in transfer_scores.items()}
        loader = DataLoader(client.dataset, batch_size=self.config.local_batch_size, shuffle=True, generator=torch.Generator().manual_seed(self.config.random_state + round_idx * 1000 + _stable_client_seed(client.client_id)), num_workers=self.config.num_workers, pin_memory=self.config.pin_memory)
        global_ref = {k: v.detach().clone().to(self.device) for k, v in global_state.items()}
        selected = set(selected_layers)
        local_model.train()
        for _ in range(self.config.local_epochs):
            for batch in loader:
                x, y = unpack_batch(move_batch_to_device(batch, self.device))
                local_model.zero_grad(set_to_none=True)
                loss = self.criterion(local_model(x), y)
                loss.backward()
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=self.config.max_grad_norm)
                with torch.no_grad():
                    for spec in self.layer_specs:
                        is_selected = spec.name in selected
                        lr_scale = 1.0 if is_selected or not self.config.ablations.use_train_gate else self.config.nonselected_lr_scale
                        mu_scale = 1.0 if is_selected or not self.config.ablations.use_train_gate else self.config.nonselected_mu_scale
                        lr = float(layer_lrs[spec.name]) * lr_scale
                        mu = float(proximal_strengths.get(spec.name, 0.0)) * mu_scale
                        if self.config.ablations.use_train_gate and round_idx >= self.config.freeze_unselected_after and not is_selected:
                            lr = 0.0
                        for pname in spec.param_names:
                            param = named_params[pname]
                            grad = torch.zeros_like(param) if param.grad is None else param.grad
                            update = grad + self.config.weight_decay * param + mu * (param - global_ref[pname])
                            param.add_(-lr * update)
        final_state = detach_state_dict(local_model)
        deltas = state_delta_by_layer(final_state, global_state, self.layer_specs)
        return {name: deltas[name] for name in selected if name in deltas}, transfer_scores, layer_lrs

    def fit(self, clients: Sequence[ClientDataset], server_val_loader: DataLoader | None = None, server_test_loader: DataLoader | None = None) -> dict[str, Any]:
        if len(clients) == 0:
            raise ValueError("At least one client dataset is required.")
        self.history = {"config": asdict(self.config), "rounds": []}
        client_weight_map = infer_client_weights(clients)
        previous_global_state = None
        previous_val_accuracy = self.evaluate(server_val_loader)["accuracy"] if server_val_loader is not None else 0.0
        previous_state = RoundState(drift=0.0, communication_ratio=0.0, validation_delta=0.0, credit_mean=0.0)
        for round_idx in range(self.config.num_rounds):
            sampled_clients = self._sample_clients(clients, round_idx)
            global_state = detach_state_dict(self.model)
            ref_sketches = self._build_reference_sketches(global_state, previous_global_state)
            if round_idx < self.config.warmup_rounds:
                action = ControllerAction(1.0, -1e9, -1)
            elif self.config.ablations.use_round_controller and self.config.controller.enabled:
                action = self.controller.choose(previous_state)
            else:
                action = ControllerAction(self.config.default_budget_fraction, self.config.default_threshold, -1)
            client_credit_dicts = []
            client_credit_details = {}
            for client in sampled_clients:
                credits, details = self._phase_a_client_credit(client, global_state, ref_sketches, round_idx)
                client_credit_dicts.append(credits)
                client_credit_details[str(client.client_id)] = details
            raw_global_credit = aggregate_global_credit(client_credit_dicts, [spec.name for spec in self.layer_specs])
            control_credit = _normalize_global_credit(raw_global_credit)
            must_include = [self.layer_specs[-1].name] if self.config.always_include_output_layer and self.layer_specs else []
            if round_idx < self.config.warmup_rounds:
                selected_layers = [spec.name for spec in self.layer_specs]
            else:
                selected_layers = select_layers_under_budget(raw_global_credit, self.layer_costs, action.budget_fraction, action.threshold, budget_scale=self.config.budget_scale, ensure_nonempty=self.config.ensure_nonempty_gate, must_include=must_include)
            layer_steps = {spec.name: float(self.config.eta_min + (self.config.eta_max - self.config.eta_min) * sigmoid(self.config.alpha_credit * control_credit[spec.name])) for spec in self.layer_specs}
            proximal_strengths = {spec.name: float(self.config.mu_min + (self.config.mu_max - self.config.mu_min) * sigmoid(-self.config.alpha_credit * control_credit[spec.name])) for spec in self.layer_specs}
            client_updates = []
            client_weights = []
            client_transfer = {}
            client_lrs = {}
            for client in sampled_clients:
                sparse_update, transfer_scores, layer_lrs = self._phase_b_client_update(client, global_state, selected_layers, proximal_strengths, round_idx)
                client_updates.append(sparse_update)
                client_weights.append(client_weight_map[client.client_id])
                client_transfer[str(client.client_id)] = transfer_scores
                client_lrs[str(client.client_id)] = layer_lrs
            aggregated_update = aggregate_sparse_updates(client_updates, client_weights, selected_layers, client_credit_dicts, self.config.ablations.use_credit_weighted_aggregation)
            momentum_update = {}
            for spec in self.layer_specs:
                if spec.name not in aggregated_update:
                    continue
                layer_payload = {}
                for pname, delta in aggregated_update[spec.name].items():
                    vel = self.server_velocity[spec.name][pname]
                    vel.mul_(self.config.aggregation_momentum).add_(delta)
                    layer_payload[pname] = vel.detach().clone()
                momentum_update[spec.name] = layer_payload
            new_state = apply_global_update(self.model, momentum_update, layer_steps)
            comm_ratio = float(sum(self.layer_costs[name] for name in selected_layers))
            client_to_server_bits = int(sum(self.layer_bits[name] for name in selected_layers) * len(sampled_clients))
            server_to_client_bits = int(self.model_bits * len(sampled_clients)) if self.config.track_server_to_client_bits else 0
            drift = 0.0
            for spec in self.layer_specs:
                before = flatten_params_from_state(global_state, spec)
                after = flatten_params_from_state(new_state, spec)
                drift += float(torch.mean((after - before) ** 2))
            val_metrics = self.evaluate(server_val_loader) if server_val_loader is not None else None
            validation_delta = float(val_metrics["accuracy"] - previous_val_accuracy) if val_metrics is not None else 0.0
            if val_metrics is not None:
                previous_val_accuracy = float(val_metrics["accuracy"])
            if self.config.ablations.use_round_controller and self.config.controller.enabled and round_idx >= self.config.warmup_rounds:
                reward = self.controller.compute_reward(validation_delta, comm_ratio, drift)
                self.controller.update(previous_state, action, reward)
            else:
                reward = validation_delta - self.config.controller.reward_comm_penalty * comm_ratio - self.config.controller.reward_drift_penalty * drift
            previous_state = RoundState(drift=drift, communication_ratio=comm_ratio, validation_delta=validation_delta, credit_mean=mean_or_zero(list(control_credit.values())))
            previous_global_state = global_state
            round_log = {
                "round": round_idx,
                "sampled_clients": [client.client_id for client in sampled_clients],
                "controller_action": {"budget_fraction": float(action.budget_fraction), "threshold": float(action.threshold), "action_index": int(action.action_index)},
                "selected_layers": selected_layers,
                "selected_layer_ratio": float(len(selected_layers) / max(len(self.layer_specs), 1)),
                "communication_ratio": comm_ratio,
                "client_to_server_bits": client_to_server_bits,
                "server_to_client_bits": server_to_client_bits,
                "total_bits": int(client_to_server_bits + server_to_client_bits),
                "drift": drift,
                "reward": reward,
                "raw_global_credit": raw_global_credit,
                "control_credit": control_credit,
                "layer_steps": layer_steps,
                "proximal_strengths": proximal_strengths,
                "client_credit_details": client_credit_details,
                "client_transfer_scores": client_transfer,
                "client_layer_lrs": client_lrs,
            }
            if val_metrics is not None:
                round_log["validation"] = val_metrics
            self.history["rounds"].append(round_log)
        if server_test_loader is not None:
            self.history["test"] = self.evaluate(server_test_loader)
        return self.history

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        return evaluate_classifier(self.model, loader, self.criterion, self.device)

    def evaluate_clients(self, clients: Sequence[ClientDataset], batch_size: int = 256) -> dict[str, object]:
        per_client = {}
        for client in clients:
            loader = DataLoader(client.dataset, batch_size=batch_size, shuffle=False)
            per_client[str(client.client_id)] = float(self.evaluate(loader)["accuracy"])
        vals = list(per_client.values())
        return {"per_client_accuracy": per_client, "mean_accuracy": float(np.mean(vals)) if vals else 0.0, "std_accuracy": float(np.std(vals)) if vals else 0.0, "worst_accuracy": float(np.min(vals)) if vals else 0.0, "p10_accuracy": float(np.percentile(np.asarray(vals, dtype=float), 10.0)) if vals else 0.0}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return torch.argmax(self.model(x.to(self.device)), dim=1).cpu()

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(x.to(self.device)), dim=1).cpu()
