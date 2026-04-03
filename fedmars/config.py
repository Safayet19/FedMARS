from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(slots=True)
class ControllerConfig:
    enabled: bool = False
    budget_candidates: Tuple[float, ...] = (0.35, 0.50, 0.70, 1.00)
    threshold_candidates: Tuple[float, ...] = (-0.05, 0.00, 0.03)
    epsilon: float = 0.15
    step_size: float = 0.25
    drift_bins: Tuple[float, ...] = (0.02, 0.10, 0.30)
    comm_bins: Tuple[float, ...] = (0.25, 0.50, 0.75)
    val_delta_bins: Tuple[float, ...] = (-0.01, 0.00, 0.01)
    credit_bins: Tuple[float, ...] = (-0.02, 0.00, 0.02)
    reward_comm_penalty: float = 0.08
    reward_drift_penalty: float = 0.03


@dataclass(slots=True)
class AblationConfig:
    use_reference_sketch: bool = True
    use_multimodal_partition: bool = True
    use_counterfactual_mixture: bool = True
    use_layer_credit: bool = True
    use_transfer_lr: bool = True
    use_round_controller: bool = False
    use_depth_weight: bool = True
    use_train_gate: bool = True
    use_credit_weighted_aggregation: bool = True


@dataclass(slots=True)
class FedMARSConfig:
    random_state: int = 42
    device: str = "cpu"

    num_rounds: int = 30
    warmup_rounds: int = 0
    positive_pair_rounds: int = 0

    client_fraction: float = 1.0
    min_clients_per_round: int = 5

    local_epochs: int = 2
    local_batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    max_grad_norm: Optional[float] = 5.0

    num_clusters: int = 3
    partition_method: str = "label"
    cluster_refresh_interval: int = 1
    max_partition_samples: int = 512
    min_examples_for_multimodal: int = 24

    mixture_conflict_beta: float = 0.35
    mixture_temperature: float = 0.40
    mixture_resolution: int = 4

    reference_momentum: float = 0.80
    reference_sketch_mode: str = "ema_unit"
    depth_weight_mode: str = "linear"

    lambda_r: float = 0.80
    lambda_c: float = 0.10
    lambda_v: float = 0.35
    
    probe_batch_size: int = 64
    probe_step: float = 0.05

    eta_min: float = 0.20
    eta_max: float = 0.90
    mu_min: float = 0.00
    mu_max: float = 0.08
    alpha_credit: float = 2.50

    rho_min: float = 0.002
    rho_max: float = 0.03
    kappa_transfer: float = 3.0
    tau_transfer: float = 0.10
    probe_batch_size: int = 64

    aggregation: str = "weighted_mean"
    ensure_nonempty_gate: bool = True
    budget_scale: int = 200
    default_budget_fraction: float = 0.70
    default_threshold: float = -0.02

    nonselected_lr_scale: float = 0.15
    nonselected_mu_scale: float = 2.00
    freeze_unselected_after: int = 2

    aggregation_momentum: float = 0.90
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0

    controller: ControllerConfig = field(default_factory=ControllerConfig)
    ablations: AblationConfig = field(default_factory=AblationConfig)