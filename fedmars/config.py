from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class AblationConfig:
    use_reference_sketch: bool = True
    use_multimodal_partition: bool = True
    use_counterfactual_mixture: bool = True
    use_layer_credit: bool = True
    use_transfer_lr: bool = True
    use_depth_weight: bool = True
    use_train_gate: bool = True
    use_credit_weighted_aggregation: bool = True


@dataclass(slots=True)
class FedMARSConfig:
    random_state: int = 42
    device: str = "cpu"

    num_rounds: int = 40
    warmup_rounds: int = 3
    positive_pair_rounds: int = 0

    client_fraction: float = 1.0
    min_clients_per_round: int = 5

    local_epochs: int = 2
    local_batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    max_grad_norm: Optional[float] = 5.0

    num_clusters: int = 3
    num_batches_per_cluster: int = 3
    transfer_probe_batches: int = 3
    partition_method: str = "label"
    cluster_refresh_interval: int = 1
    max_partition_samples: int = 512
    min_examples_for_multimodal: int = 24

    mixture_conflict_beta: float = 0.20
    mixture_temperature: float = 0.60
    mixture_resolution: int = 4
    mixture_steps: int = 40

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

    aggregation: str = "weighted_mean"
    ensure_nonempty_gate: bool = True
    always_include_output_layer: bool = True
    budget_scale: int = 200
    default_budget_fraction: float = 1.00
    default_threshold: float = -0.50

    nonselected_lr_scale: float = 0.40
    nonselected_mu_scale: float = 1.40
    freeze_unselected_after: int = 999

    aggregation_momentum: float = 0.90
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0

    param_bits: int = 32
    track_server_to_client_bits: bool = True

    ablations: AblationConfig = field(default_factory=AblationConfig)
