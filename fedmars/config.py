from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class FedMARSConfig:
    random_state: int = 42
    device: str = "auto"

    num_rounds: int = 200
    warmup_rounds: int = 5
    client_fraction: float = 0.10
    min_clients_per_round: int = 10

    local_epochs: int = 2
    local_batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    max_grad_norm: Optional[float] = 5.0

    num_clusters: int = 3
    num_batches_per_cluster: int = 3
    transfer_probe_batches: int = 3
    partition_method: str = "label"
    max_partition_samples: int = 1024
    min_examples_for_multimodal: int = 24

    mixture_conflict_beta: float = 0.20
    mixture_temperature: float = 0.60
    mixture_entropy: float = 0.048
    mixture_steps: int = 50

    reference_momentum: float = 0.80
    reference_sketch_mode: str = "ema_unit"
    depth_weight_mode: str = "linear"

    lambda_r: float = 0.35
    lambda_c: float = 0.02
    lambda_v: float = 0.30

    probe_batch_size: int = 64
    probe_step: float = 0.05

    eta_min: float = 0.50
    eta_max: float = 1.00

    mu_min: float = 0.00
    mu_max: float = 0.03

    alpha_credit: float = 1.80

    rho_min: float = 0.002
    rho_max: float = 0.03
    kappa_transfer: float = 3.0
    tau_transfer: float = 0.10

    ensure_nonempty_gate: bool = True
    default_budget_fraction: float = 0.35
    default_threshold: float = -0.50

    nonselected_lr_scale: float = 0.40
    nonselected_mu_scale: float = 1.40

    aggregation_momentum: float = 0.90
    credit_weight_gamma: float = 1.00
    delta_clip_factor: float = 2.50

    weight_decay: float = 1e-4
    label_smoothing: float = 0.0

    param_bits: int = 32
    track_server_to_client_bits: bool = True