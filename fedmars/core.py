from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .aggregation import aggregate_sparse_updates, apply_global_update, apply_layer_momentum, select_layers_under_budget
from .config import FedMARSConfig
from .controller import AdaptiveRoundController, ControllerAction, RoundState
from .credit import aggregate_global_credit, build_reference_sketch, compute_layer_credit, postprocess_control_credit
from .data import ClientDataset, infer_client_weights
from .layers import (
    LayerSpec,
    build_layer_specs,
    compute_depth_weights,
    compute_layer_bits,
    compute_layer_costs,
    flatten_grads_from_model,
    flatten_params_from_state,
    layer_name_to_spec,
    state_delta_by_layer,
)
from .mixture import select_counterfactual_mixture, weighted_conflict
from .partition import build_local_modes, sample_batches_from_indices, sample_probe_batches
from .utils import (
    clone_model,
    detach_state_dict,
    evaluate_classifier,
    load_state_dict_,
    mean_or_zero,
    move_batch_to_device,
    percentile,
    safe_cosine,
    set_seed,
    sigmoid,
    unpack_batch,
)


def _stable_client_seed(value: int | str) -> int:
    s = str(value)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(s)) % 1000003


def _unflatten_layer_vector(
    vector: torch.Tensor,
    spec: LayerSpec,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    offset = 0
    vec = vector.reshape(-1)
    for pname in spec.param_names:
        ref = state_dict[pname]
        numel = int(ref.numel())
        chunk = vec[offset : offset + numel].reshape(ref.shape).to(dtype=ref.dtype)
        out[pname] = chunk
        offset += numel
    return out


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
        self.layer_bits = compute_layer_bits(self.layer_specs, param_bits=self.config.param_bits)
        self.total_model_bits = int(sum(self.layer_bits.values()))
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
        self.cluster_cache: dict[str, list[list[int]]] = {}
        self.history: dict[str, Any] = {
            "config": asdict(self.config),
            "layer_specs": [asdict(spec) for spec in self.layer_specs],
            "layer_costs": self.layer_costs,
            "layer_bits": self.layer_bits,
            "rounds": [],
        }

    def _sample_clients(self, clients: Sequence[ClientDataset], round_idx: int) -> list[ClientDataset]:
        rng = np.random.default_rng(self.config.random_state + round_idx)
        num_clients = len(clients)
        choose = max(self.config.min_clients_per_round, int(round(self.config.client_fraction * num_clients)))
        choose = min(max(1, choose), num_clients)
        picked = rng.choice(np.arange(num_clients), size=choose, replace=False)
        return [clients[int(i)] for i in picked.tolist()]

    def _make_loader(self, dataset, seed: int, shuffle: bool) -> DataLoader:
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=self.config.local_batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def _get_client_groups(self, client: ClientDataset, round_idx: int) -> list[list[int]]:
        if not self.config.ablations.use_multimodal_partition:
            return [list(range(len(client.dataset)))]
        refresh_interval = max(1, self.config.cluster_refresh_interval)
        key = str(client.client_id)
        if key in self.cluster_cache and round_idx % refresh_interval != 0:
            return self.cluster_cache[key]
        groups = build_local_modes(
            dataset=client.dataset,
            num_clusters=self.config.num_clusters,
            method=self.config.partition_method,
            seed=self.config.random_state + round_idx * 1000 + _stable_client_seed(client.client_id),
            max_samples=self.config.max_partition_samples,
            min_examples_for_multimodal=self.config.min_examples_for_multimodal,
        )
        self.cluster_cache[key] = groups
        return groups

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

        use_ema = self.config.reference_sketch_mode.startswith("ema_")

        for spec in self.layer_specs:
            cur = flatten_params_from_state(current_state, spec)
            prev = flatten_params_from_state(previous_state, spec)
            raw_delta = cur - prev
            sketch = build_reference_sketch(
                raw_delta,
                mode=self.config.reference_sketch_mode,
                topk_fraction=self.config.reference_topk_fraction,
            )
            if sketch is None:
                refs[spec.name] = self.ref_memory[spec.name]
                continue
            if use_ema:
                prev_ref = self.ref_memory[spec.name]
                if prev_ref is None:
                    merged = sketch
                else:
                    merged = float(self.config.reference_momentum) * prev_ref + (1.0 - float(self.config.reference_momentum)) * sketch
                    merged = merged / (torch.norm(merged) + 1e-12)
                refs[spec.name] = merged
            else:
                refs[spec.name] = sketch

        self.ref_memory = {k: (v.detach().clone() if v is not None else None) for k, v in refs.items()}
        return refs

    def _compute_batch_loss(self, model: torch.nn.Module, batch) -> float:
        model.eval()
        with torch.no_grad():
            batch = move_batch_to_device(batch, self.device)
            x, y = unpack_batch(batch)
            logits = model(x)
            loss = self.criterion(logits, y)
        return float(loss)

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
        num_batches = max(2, int(self.config.transfer_probe_batches))
        batches = sample_probe_batches(
            client.dataset,
            batch_size=self.config.probe_batch_size,
            num_batches=num_batches,
            seed_start=round_seed,
        )

        probe_model = clone_model(self.model, self.device)
        load_state_dict_(probe_model, global_state)
        batch_grads = [self._compute_batch_layer_grads(probe_model, batch) for batch in batches]

        out: dict[str, float] = {}
        for spec in self.layer_specs:
            pair_scores: list[float] = []
            for i in range(len(batch_grads)):
                for j in range(i + 1, len(batch_grads)):
                    pair_scores.append(safe_cosine(batch_grads[i][spec.name], batch_grads[j][spec.name]))
            out[spec.name] = mean_or_zero(pair_scores)
        return out

    def _map_transfer_to_lr(self, transfer_score: float) -> float:
        if not self.config.ablations.use_transfer_lr:
            return float(self.config.rho_max)
        return float(
            self.config.rho_min
            + (self.config.rho_max - self.config.rho_min)
            * sigmoid(self.config.kappa_transfer * (transfer_score - self.config.tau_transfer))
        )

    def _probe_layer_gain(
        self,
        global_state: Mapping[str, torch.Tensor],
        spec: LayerSpec,
        mixed_gradient: torch.Tensor,
        probe_batch,
    ) -> float:
        probe_step = float(getattr(self.config, "probe_step", 0.0))
        if probe_step <= 0.0:
            return 0.0

        probe_model = clone_model(self.model, self.device)
        load_state_dict_(probe_model, global_state)
        loss_before = self._compute_batch_loss(probe_model, probe_batch)
        step_chunks = _unflatten_layer_vector(mixed_gradient, spec, global_state)

        with torch.no_grad():
            named_params = dict(probe_model.named_parameters())
            for pname in spec.param_names:
                named_params[pname].add_(-probe_step * step_chunks[pname].to(named_params[pname].device))

        loss_after = self._compute_batch_loss(probe_model, probe_batch)
        return float(loss_before - loss_after)

    def _estimate_cluster_gradients(
        self,
        client: ClientDataset,
        global_state: Mapping[str, torch.Tensor],
        round_idx: int,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, dict[str, dict[str, float]]]]:
        groups = self._get_client_groups(client, round_idx)
        probe_model = clone_model(self.model, self.device)
        load_state_dict_(probe_model, global_state)

        cluster_grads: list[dict[str, torch.Tensor]] = []
        cluster_details: dict[str, dict[str, dict[str, float]]] = {}

        client_seed = _stable_client_seed(client.client_id)

        for group_idx, indices in enumerate(groups):
            batches = sample_batches_from_indices(
                dataset=client.dataset,
                indices=indices,
                batch_size=self.config.local_batch_size,
                num_batches=self.config.num_batches_per_cluster,
                seed_start=self.config.random_state + round_idx * 10000 + group_idx * 100 + client_seed,
            )
            batch_grads = [self._compute_batch_layer_grads(probe_model, batch) for batch in batches]
            cluster_grad: dict[str, torch.Tensor] = {}
            layer_details: dict[str, dict[str, float]] = {}
            for spec in self.layer_specs:
                grads = [g[spec.name] for g in batch_grads]
                stacked = torch.stack(grads, dim=0)
                cluster_grad[spec.name] = stacked.mean(dim=0)
                pair_scores: list[float] = []
                if len(grads) > 1:
                    for i in range(len(grads)):
                        for j in range(i + 1, len(grads)):
                            pair_scores.append(safe_cosine(grads[i], grads[j]))
                layer_details[spec.name] = {
                    "batch_stability": mean_or_zero(pair_scores) if pair_scores else 1.0,
                    "cluster_gradient_norm": float(torch.norm(cluster_grad[spec.name])),
                }
            cluster_grads.append(cluster_grad)
            cluster_details[str(group_idx)] = layer_details

        return cluster_grads, cluster_details

    def _phase_a_client_credit(
        self,
        client: ClientDataset,
        global_state: Mapping[str, torch.Tensor],
        ref_sketches: Mapping[str, torch.Tensor | None],
        round_idx: int,
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        cluster_grads, cluster_details = self._estimate_cluster_gradients(client, global_state, round_idx)

        credits: dict[str, float] = {}
        details: dict[str, dict[str, Any]] = {}

        probe_batch = None
        if float(getattr(self.config, "lambda_v", 0.0)) > 0.0:
            probe_batches = sample_probe_batches(
                client.dataset,
                batch_size=self.config.probe_batch_size,
                num_batches=1,
                seed_start=self.config.random_state + round_idx * 20000 + _stable_client_seed(client.client_id),
            )
            probe_batch = probe_batches[0]

        for spec in self.layer_specs:
            grads = [cg[spec.name] for cg in cluster_grads]
            reference = ref_sketches.get(spec.name) if self.config.ablations.use_reference_sketch else None

            if self.config.ablations.use_counterfactual_mixture:
                weights, mixed, conflict, objective = select_counterfactual_mixture(
                    gradients=grads,
                    reference=reference,
                    beta=self.config.mixture_conflict_beta,
                    temperature=self.config.mixture_temperature,
                    steps=self.config.mixture_steps,
                )
            else:
                weights = np.full(len(grads), 1.0 / max(1, len(grads)), dtype=np.float32)
                mixed = sum(grads) / max(1, len(grads))
                conflict = weighted_conflict(weights, grads)
                objective = float(torch.norm(mixed))

            probe_gain = 0.0
            if probe_batch is not None:
                probe_gain = self._probe_layer_gain(global_state, spec, mixed, probe_batch)

            record = compute_layer_credit(
                reference=reference,
                mixed_gradient=mixed,
                residual_conflict=conflict if self.config.ablations.use_layer_credit else 0.0,
                cost=self.layer_costs[spec.name] if self.config.ablations.use_layer_credit else 0.0,
                depth_weight=self.depth_weights[spec.name] if self.config.ablations.use_depth_weight else 1.0,
                lambda_r=self.config.lambda_r if self.config.ablations.use_layer_credit else 0.0,
                lambda_c=self.config.lambda_c if self.config.ablations.use_layer_credit else 0.0,
                probe_gain=probe_gain if self.config.ablations.use_layer_credit else 0.0,
                lambda_v=self.config.lambda_v if self.config.ablations.use_layer_credit else 0.0,
            )

            credit_value = record.credit if self.config.ablations.use_layer_credit else record.benefit
            cluster_stabilities = [
                float(cluster_details[str(group_idx)][spec.name]["batch_stability"])
                for group_idx in range(len(cluster_grads))
            ]

            credits[spec.name] = float(credit_value)
            details[spec.name] = {
                "cluster_count": len(grads),
                "batches_per_cluster": int(self.config.num_batches_per_cluster),
                "mixture_weights": weights.tolist(),
                "objective": float(objective),
                "benefit": float(record.benefit),
                "risk": float(record.risk),
                "cost": float(record.cost),
                "depth_weight": float(record.depth_weight),
                "alignment": float(record.alignment),
                "gradient_norm": float(record.gradient_norm),
                "credit": float(credit_value),
                "probe_gain": float(record.probe_gain),
                "cluster_batch_stability": cluster_stabilities,
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
            round_seed=self.config.random_state + round_idx * 1000 + 500000 + _stable_client_seed(client.client_id),
        )
        layer_lrs = {name: self._map_transfer_to_lr(score) for name, score in transfer_scores.items()}

        loader = self._make_loader(
            client.dataset,
            seed=self.config.random_state + round_idx * 1000 + 700000 + _stable_client_seed(client.client_id),
            shuffle=True,
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
                        lr_scale = 1.0
                        mu_scale = 1.0

                        if self.config.ablations.use_train_gate:
                            if is_selected:
                                lr_scale = 1.0
                                mu_scale = 1.0
                            else:
                                lr_scale = self.config.nonselected_lr_scale
                                mu_scale = self.config.nonselected_mu_scale
                                if round_idx >= self.config.freeze_unselected_after:
                                    lr_scale = 0.0

                        lr = float(layer_lrs[spec.name]) * float(lr_scale)
                        mu = float(proximal_strengths.get(spec.name, 0.0)) * float(mu_scale)

                        for pname in spec.param_names:
                            param = named_params[pname]
                            grad = torch.zeros_like(param) if param.grad is None else param.grad
                            update = grad + self.config.weight_decay * param + mu * (param - global_ref[pname])
                            param.add_(-lr * update)

        final_state = detach_state_dict(local_model)
        deltas = state_delta_by_layer(final_state, global_state, self.layer_specs)
        sparse = {name: deltas[name] for name in selected if name in deltas}
        return sparse, transfer_scores, layer_lrs

    def _reference_bits_per_client(self, ref_sketches: Mapping[str, torch.Tensor | None]) -> int:
        base_mode = self.config.reference_sketch_mode[4:] if self.config.reference_sketch_mode.startswith("ema_") else self.config.reference_sketch_mode
        total = 0
        for spec in self.layer_specs:
            if ref_sketches.get(spec.name) is None:
                continue
            if base_mode == "unit":
                total += self.layer_bits[spec.name]
            elif base_mode == "sign":
                total += int(spec.numel)
            elif base_mode == "topk_sign":
                k = max(1, int(np.ceil(float(self.config.reference_topk_fraction) * spec.numel)))
                index_bits = max(1, int(np.ceil(np.log2(max(spec.numel, 2)))))
                total += int(k * (1 + index_bits))
            else:
                total += self.layer_bits[spec.name]
        return int(total)

    def _compute_round_bits(
        self,
        sampled_clients: Sequence[ClientDataset],
        ref_sketches: Mapping[str, torch.Tensor | None],
        sparse_updates: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
        selected_layers: Sequence[str],
    ) -> dict[str, float]:
        client_to_server_bits = 0
        for update in sparse_updates:
            for layer_name in update.keys():
                client_to_server_bits += int(self.layer_bits.get(layer_name, 0))

        if self.config.track_server_to_client_bits:
            server_to_client_bits = int(len(sampled_clients) * self.total_model_bits)
            server_to_client_bits += int(len(sampled_clients) * self._reference_bits_per_client(ref_sketches))
            server_to_client_bits += int(len(sampled_clients) * 64)
        else:
            server_to_client_bits = 0

        communication_ratio = float(sum(self.layer_costs.get(name, 0.0) for name in selected_layers))
        return {
            "communication_ratio": communication_ratio,
            "client_to_server_bits": int(client_to_server_bits),
            "server_to_client_bits": int(server_to_client_bits),
            "total_bits": int(client_to_server_bits + server_to_client_bits),
        }

    def _compute_drift(
        self,
        before_state: Mapping[str, torch.Tensor],
        after_state: Mapping[str, torch.Tensor],
    ) -> float:
        total = 0.0
        denom = 0
        for spec in self.layer_specs:
            before = flatten_params_from_state(before_state, spec)
            after = flatten_params_from_state(after_state, spec)
            total += float(torch.mean((after - before) ** 2))
            denom += 1
        return float(total / max(denom, 1))

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
        previous_round_state = RoundState(drift=0.0, communication_ratio=self.config.default_budget_fraction, validation_delta=0.0, credit_mean=0.0)

        if server_val_loader is not None:
            previous_val_accuracy = self.evaluate(server_val_loader)["accuracy"]

        for round_idx in range(self.config.num_rounds):
            sampled_clients = self._sample_clients(clients, round_idx)
            global_state = detach_state_dict(self.model)
            ref_sketches = self._build_reference_sketches(global_state, previous_global_state)

            if (
                self.config.ablations.use_round_controller
                and self.config.controller.enabled
                and round_idx >= self.config.warmup_rounds
            ):
                action = self.controller.choose(previous_round_state)
            else:
                action = ControllerAction(
                    budget_fraction=self.config.default_budget_fraction,
                    threshold=self.config.default_threshold,
                    action_index=-1,
                )

            client_credit_dicts: list[dict[str, float]] = []
            client_credit_details: dict[str, dict[str, dict[str, Any]]] = {}

            for client in sampled_clients:
                credits, details = self._phase_a_client_credit(client, global_state, ref_sketches, round_idx)
                client_credit_dicts.append(credits)
                client_credit_details[str(client.client_id)] = details

            raw_global_credit = aggregate_global_credit(
                client_credit_dicts=client_credit_dicts,
                layer_names=[spec.name for spec in self.layer_specs],
                method=self.config.global_credit_aggregator,
            )
            control_credit = postprocess_control_credit(
                raw_credit=raw_global_credit,
                mode=self.config.control_credit_mode,
                clip=self.config.control_credit_clip,
            )

            selected_layers = select_layers_under_budget(
                global_credit=control_credit,
                layer_costs=self.layer_costs,
                budget_fraction=action.budget_fraction,
                threshold=action.threshold,
                ensure_nonempty=self.config.ensure_nonempty_gate,
            )

            layer_steps = {
                spec.name: float(
                    self.config.eta_min
                    + (self.config.eta_max - self.config.eta_min)
                    * sigmoid(self.config.alpha_credit * control_credit[spec.name])
                )
                for spec in self.layer_specs
            }

            proximal_strengths = {
                spec.name: float(
                    self.config.mu_min
                    + (self.config.mu_max - self.config.mu_min)
                    * sigmoid(-self.config.alpha_credit * control_credit[spec.name])
                )
                for spec in self.layer_specs
            }

            client_updates: list[dict[str, dict[str, torch.Tensor]]] = []
            client_weights: list[float] = []
            client_transfer: dict[str, dict[str, float]] = {}
            client_lrs: dict[str, dict[str, float]] = {}

            for client in sampled_clients:
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
                mode=self.config.aggregation,
            )

            momentum_update = apply_layer_momentum(
                aggregated_updates=aggregated_update,
                velocity_state=self.server_velocity,
                momentum=self.config.aggregation_momentum,
            )
            new_state = apply_global_update(self.model, momentum_update, layer_steps)

            bit_stats = self._compute_round_bits(
                sampled_clients=sampled_clients,
                ref_sketches=ref_sketches,
                sparse_updates=client_updates,
                selected_layers=selected_layers,
            )
            drift = self._compute_drift(global_state, new_state)

            val_metrics = None
            validation_delta = 0.0
            if server_val_loader is not None:
                val_metrics = self.evaluate(server_val_loader)
                validation_delta = float(val_metrics["accuracy"] - previous_val_accuracy)
                previous_val_accuracy = float(val_metrics["accuracy"])

            if (
                self.config.ablations.use_round_controller
                and self.config.controller.enabled
                and round_idx >= self.config.warmup_rounds
            ):
                reward = self.controller.compute_reward(validation_delta, bit_stats["communication_ratio"], drift)
                self.controller.update(previous_round_state, action, reward)
            else:
                reward = (
                    validation_delta
                    - self.config.controller.reward_comm_penalty * bit_stats["communication_ratio"]
                    - self.config.controller.reward_drift_penalty * drift
                )

            previous_round_state = RoundState(
                drift=drift,
                communication_ratio=bit_stats["communication_ratio"],
                validation_delta=validation_delta,
                credit_mean=mean_or_zero(list(control_credit.values())),
            )
            previous_global_state = global_state

            round_log = {
                "round": round_idx,
                "sampled_clients": [client.client_id for client in sampled_clients],
                "action": {
                    "budget_fraction": float(action.budget_fraction),
                    "threshold": float(action.threshold),
                    "action_index": int(action.action_index),
                },
                "selected_layers": selected_layers,
                "communication_ratio": float(bit_stats["communication_ratio"]),
                "client_to_server_bits": int(bit_stats["client_to_server_bits"]),
                "server_to_client_bits": int(bit_stats["server_to_client_bits"]),
                "total_bits": int(bit_stats["total_bits"]),
                "drift": float(drift),
                "reward": float(reward),
                "raw_global_credit": raw_global_credit,
                "global_credit": control_credit,
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
