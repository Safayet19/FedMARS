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
from .layers import (
    LayerSpec,
    build_layer_specs,
    compute_depth_weights,
    compute_layer_costs,
    flatten_grads_from_model,
    flatten_params_from_state,
    layer_name_to_spec,
    state_delta_by_layer,
)
from .mixture import select_counterfactual_mixture
from .partition import build_local_modes, sample_batch_from_indices, sample_probe_batches
from .utils import (
    clone_model,
    detach_state_dict,
    evaluate_classifier,
    load_state_dict_,
    move_batch_to_device,
    safe_cosine,
    set_seed,
    sigmoid,
    unpack_batch,
)


def _stable_client_seed(value: int | str) -> int:
    s = str(value)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(s)) % 1000003


def _mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


class FedMARS:
    def __init__(
        self,
        model: torch.nn.Module,
        config: FedMARSConfig | None = None,
        criterion: torch.nn.Module | None = None,
    ):
        self.config = config if config is not None else FedMARSConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.criterion = (
            criterion
            if criterion is not None
            else torch.nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        )

        set_seed(self.config.random_state)

        self.layer_specs: list[LayerSpec] = build_layer_specs(self.model)
        self.layer_map = layer_name_to_spec(self.layer_specs)
        self.depth_weights = compute_depth_weights(self.layer_specs, mode=self.config.depth_weight_mode)
        self.layer_costs = compute_layer_costs(self.layer_specs)
        self.controller = AdaptiveRoundController(self.config.controller, seed=self.config.random_state)

        named_params = dict(self.model.named_parameters())
        self.ref_memory: dict[str, torch.Tensor | None] = {spec.name: None for spec in self.layer_specs}
        self.server_velocity: dict[str, dict[str, torch.Tensor]] = {
            spec.name: {
                pname: torch.zeros_like(named_params[pname]).detach().clone().cpu()
                for pname in spec.param_names
            }
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

    def _build_reference_sketches(
        self,
        current_state: Mapping[str, torch.Tensor],
        previous_state: Mapping[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor | None]:
        refs: dict[str, torch.Tensor | None] = {}

        if previous_state is None:
            refs = {spec.name: None for spec in self.layer_specs}
            self.ref_memory = {spec.name: None for spec in self.layer_specs}
            return refs

        for spec in self.layer_specs:
            cur = flatten_params_from_state(current_state, spec)
            prev = flatten_params_from_state(previous_state, spec)
            raw_delta = cur - prev

            unit = build_reference_sketch(raw_delta, mode=self.config.reference_sketch_mode)
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

    def _compute_batch_layer_grads(self, model: torch.nn.Module, batch) -> dict[str, torch.Tensor]:
        model.train()
        model.zero_grad(set_to_none=True)
        batch = move_batch_to_device(batch, self.device)
        x, y = unpack_batch(batch)
        logits = model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        return {spec.name: flatten_grads_from_model(model, spec).detach().cpu() for spec in self.layer_specs}

    def _compute_transfer_scores(
        self,
        client: ClientDataset,
        global_state: Mapping[str, torch.Tensor],
        round_seed: int,
    ) -> dict[str, float]:
        probe1, probe2 = sample_probe_batches(client.dataset, self.config.probe_batch_size, seed=round_seed)

        probe_model = clone_model(self.model, self.device)
        load_state_dict_(probe_model, global_state)
        grads1 = self._compute_batch_layer_grads(probe_model, probe1)

        load_state_dict_(probe_model, global_state)
        grads2 = self._compute_batch_layer_grads(probe_model, probe2)

        return {spec.name: safe_cosine(grads1[spec.name], grads2[spec.name]) for spec in self.layer_specs}

    def _map_transfer_to_lr(self, transfer_score: float) -> float:
        if not self.config.ablations.use_transfer_lr:
            return float(self.config.rho_max)
        return float(
            self.config.rho_min
            + (self.config.rho_max - self.config.rho_min)
            * sigmoid(self.config.kappa_transfer * (transfer_score - self.config.tau_transfer))
        )

    def _phase_a_client_credit(
        self,
        client: ClientDataset,
        global_state: Mapping[str, torch.Tensor],
        ref_sketches: Mapping[str, torch.Tensor | None],
        round_idx: int,
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        client_seed = _stable_client_seed(client.client_id)

        if self.config.ablations.use_multimodal_partition:
            groups = build_local_modes(
                dataset=client.dataset,
                num_clusters=self.config.num_clusters,
                method=self.config.partition_method,
                seed=self.config.random_state + round_idx * 1000 + client_seed,
                max_samples=self.config.max_partition_samples,
                min_examples_for_multimodal=self.config.min_examples_for_multimodal,
            )
        else:
            groups = [list(range(len(client.dataset)))]

        cluster_grads: list[dict[str, torch.Tensor]] = []
        for group_idx, indices in enumerate(groups):
            batch = sample_batch_from_indices(
                dataset=client.dataset,
                indices=indices,
                batch_size=self.config.local_batch_size,
                seed=self.config.random_state + round_idx * 10000 + group_idx + client_seed,
            )
            probe_model = clone_model(self.model, self.device)
            load_state_dict_(probe_model, global_state)
            cluster_grads.append(self._compute_batch_layer_grads(probe_model, batch))

        credits: dict[str, float] = {}
        details: dict[str, dict[str, Any]] = {}

        for spec in self.layer_specs:
            grads = [cg[spec.name] for cg in cluster_grads]
            reference = ref_sketches.get(spec.name) if self.config.ablations.use_reference_sketch else None

            if self.config.ablations.use_counterfactual_mixture:
                weights, mixed, conflict, objective = select_counterfactual_mixture(
                    grads,
                    reference,
                    self.config.mixture_conflict_beta,
                    self.config.mixture_temperature,
                )
            else:
                weights = np.ones(len(grads), dtype=float) / max(len(grads), 1)
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
            )

            credits[spec.name] = record.credit if self.config.ablations.use_layer_credit else record.benefit
            details[spec.name] = {
                "weights": weights.tolist(),
                "conflict": float(conflict),
                "objective": float(objective),
                "credit": float(credits[spec.name]),
                "benefit": float(record.benefit),
                "alignment": float(record.alignment),
                "norm": float(record.norm),
            }

        return credits, details

    def _phase_b_client_update(
        self,
        client: ClientDataset,
        global_state: Mapping[str, torch.Tensor],
        selected_layers: Sequence[str],
        proximal_strengths: Mapping[str, float],
        round_idx: int,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float], dict[str, float]]:
        local_model = clone_model(self.model, self.device)
        load_state_dict_(local_model, global_state)
        named_params = dict(local_model.named_parameters())

        transfer_scores = self._compute_transfer_scores(
            client,
            global_state,
            round_seed=self.config.random_state + 30000 + round_idx + _stable_client_seed(client.client_id),
        )
        layer_lrs = {name: self._map_transfer_to_lr(score) for name, score in transfer_scores.items()}

        loader = DataLoader(
            client.dataset,
            batch_size=self.config.local_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        global_ref = {k: v.detach().clone().to(self.device) for k, v in global_state.items()}
        selected = set(selected_layers)

        local_model.train()
        for _ in range(self.config.local_epochs):
            for batch in loader:
                batch = move_batch_to_device(batch, self.device)
                x, y = unpack_batch(batch)

                local_model.zero_grad(set_to_none=True)
                loss = self.criterion(local_model(x), y)
                loss.backward()

                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=self.config.max_grad_norm)

                with torch.no_grad():
                    for spec in self.layer_specs:
                        is_selected = spec.name in selected

                        if self.config.ablations.use_train_gate:
                            lr_scale = 1.0 if is_selected else self.config.nonselected_lr_scale
                            mu_scale = 1.0 if is_selected else self.config.nonselected_mu_scale
                        else:
                            lr_scale = 1.0
                            mu_scale = 1.0

                        lr = float(layer_lrs[spec.name]) * lr_scale
                        mu = float(proximal_strengths.get(spec.name, 0.0)) * mu_scale

                        if (
                            self.config.ablations.use_train_gate
                            and round_idx >= self.config.freeze_unselected_after
                            and not is_selected
                        ):
                            lr = 0.0

                        for pname in spec.param_names:
                            param = named_params[pname]
                            grad = torch.zeros_like(param) if param.grad is None else param.grad
                            update = grad + self.config.weight_decay * param + mu * (param - global_ref[pname])
                            param.add_(-lr * update)

        final_state = detach_state_dict(local_model)
        deltas = state_delta_by_layer(final_state, global_state, self.layer_specs)
        sparse = {name: deltas[name] for name in selected if name in deltas}
        return sparse, transfer_scores, layer_lrs

    def fit(
        self,
        clients: Sequence[ClientDataset],
        server_val_loader: DataLoader | None = None,
        server_test_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        if len(clients) == 0:
            raise ValueError("At least one client dataset is required.")

        client_weight_map = infer_client_weights(clients)
        previous_global_state: Mapping[str, torch.Tensor] | None = None
        previous_val_accuracy = 0.0
        previous_state = RoundState(drift=0.0, communication_ratio=0.0, validation_delta=0.0, credit_mean=0.0)

        if server_val_loader is not None:
            previous_val_accuracy = self.evaluate(server_val_loader)["accuracy"]

        for round_idx in range(self.config.num_rounds):
            sampled_clients = self._sample_clients(clients, round_idx)
            global_state = detach_state_dict(self.model)
            ref_sketches = self._build_reference_sketches(global_state, previous_global_state)

            if self.config.ablations.use_round_controller and self.config.controller.enabled:
                action = self.controller.choose(previous_state)
            else:
                action = ControllerAction(
                    self.config.default_budget_fraction,
                    self.config.default_threshold,
                    -1,
                )

            client_credit_dicts: list[dict[str, float]] = []
            client_credit_details: dict[str, dict[str, dict[str, Any]]] = {}

            for client in sampled_clients:
                credits, details = self._phase_a_client_credit(client, global_state, ref_sketches, round_idx)
                client_credit_dicts.append(credits)
                client_credit_details[str(client.client_id)] = details

            global_credit = aggregate_global_credit(
                client_credit_dicts,
                [spec.name for spec in self.layer_specs],
            )

            selected_layers = select_layers_under_budget(
                global_credit=global_credit,
                layer_costs=self.layer_costs,
                budget_fraction=action.budget_fraction,
                threshold=action.threshold,
                budget_scale=self.config.budget_scale,
                ensure_nonempty=self.config.ensure_nonempty_gate,
            )

            layer_steps = {
                spec.name: float(
                    self.config.eta_min
                    + (self.config.eta_max - self.config.eta_min)
                    * sigmoid(self.config.alpha_credit * global_credit[spec.name])
                )
                for spec in self.layer_specs
            }

            proximal_strengths = {
                spec.name: float(
                    self.config.mu_min
                    + (self.config.mu_max - self.config.mu_min)
                    * sigmoid(-self.config.alpha_credit * global_credit[spec.name])
                )
                for spec in self.layer_specs
            }

            client_updates: list[dict[str, dict[str, torch.Tensor]]] = []
            client_weights: list[float] = []
            client_transfer: dict[str, dict[str, float]] = {}
            client_lrs: dict[str, dict[str, float]] = {}

            for client, credit_dict in zip(sampled_clients, client_credit_dicts):
                sparse_update, transfer_scores, layer_lrs = self._phase_b_client_update(
                    client=client,
                    global_state=global_state,
                    selected_layers=selected_layers,
                    proximal_strengths=proximal_strengths,
                    round_idx=round_idx,
                )
                client_updates.append(sparse_update)
                client_weights.append(client_weight_map[client.client_id])
                client_transfer[str(client.client_id)] = transfer_scores
                client_lrs[str(client.client_id)] = layer_lrs

            aggregated_update = aggregate_sparse_updates(
                sparse_updates=client_updates,
                client_weights=client_weights,
                selected_layers=selected_layers,
                client_credit_dicts=client_credit_dicts,
                use_credit_weighting=self.config.ablations.use_credit_weighted_aggregation,
            )

            momentum_update: dict[str, dict[str, torch.Tensor]] = {}
            for spec in self.layer_specs:
                if spec.name not in aggregated_update:
                    continue
                layer_payload: dict[str, torch.Tensor] = {}
                for pname, delta in aggregated_update[spec.name].items():
                    vel = self.server_velocity[spec.name][pname]
                    vel.mul_(self.config.aggregation_momentum).add_(delta)
                    layer_payload[pname] = vel.detach().clone()
                momentum_update[spec.name] = layer_payload

            new_state = apply_global_update(self.model, momentum_update, layer_steps)

            comm_ratio = float(sum(self.layer_costs[name] for name in selected_layers))
            drift = 0.0
            for spec in self.layer_specs:
                before = flatten_params_from_state(global_state, spec)
                after = flatten_params_from_state(new_state, spec)
                drift += float(torch.mean((after - before) ** 2))

            val_metrics = None
            validation_delta = 0.0
            if server_val_loader is not None:
                val_metrics = self.evaluate(server_val_loader)
                validation_delta = float(val_metrics["accuracy"] - previous_val_accuracy)
                previous_val_accuracy = float(val_metrics["accuracy"])

            if self.config.ablations.use_round_controller and self.config.controller.enabled:
                reward = self.controller.compute_reward(validation_delta, comm_ratio, drift)
                self.controller.update(previous_state, action, reward)
            else:
                reward = (
                    validation_delta
                    - self.config.controller.reward_comm_penalty * comm_ratio
                    - self.config.controller.reward_drift_penalty * drift
                )

            previous_state = RoundState(
                drift=drift,
                communication_ratio=comm_ratio,
                validation_delta=validation_delta,
                credit_mean=_mean_or_zero(list(global_credit.values())),
            )
            previous_global_state = global_state

            round_log = {
                "round": round_idx,
                "sampled_clients": [client.client_id for client in sampled_clients],
                "selected_layers": selected_layers,
                "communication_ratio": comm_ratio,
                "drift": drift,
                "reward": reward,
                "global_credit": global_credit,
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

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x.to(self.device))
            return torch.argmax(logits, dim=1).cpu()

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x.to(self.device))
            return torch.softmax(logits, dim=1).cpu()