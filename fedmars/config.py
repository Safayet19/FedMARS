from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(slots=True)
class ControllerConfig:
    enabled: bool = True
    budget_candidates: Tuple[float, ...] = (0.35, 0.50, 0.70, 1.00)
    threshold_candidates: Tuple[float, ...] = (-0.05, 0.00, 0.05)
    epsilon: float = 0.10
    step_size: float = 0.25
    drift_bins: Tuple[float, ...] = (0.02, 0.10, 0.30)
    comm_bins: Tuple[float, ...] = (0.25, 0.50, 0.75)
    val_delta_bins: Tuple[float, ...] = (-0.01, 0.00, 0.01)
    credit_bins: Tuple[float, ...] = (-0.05, 0.00, 0.05)
    reward_comm_penalty: float = 0.20
    reward_drift_penalty: float = 0.20


@dataclass(slots=True)
class AblationConfig:
    use_reference_sketch: bool = True
    use_multimodal_partition: bool = True
    use_counterfactual_mixture: bool = True
    use_layer_credit: bool = True
    use_transfer_lr: bool = True
    use_round_controller: bool = True
    use_depth_weight: bool = True


@dataclass(slots=True)
class FedMARSConfig:
    random_state: int = 42
    device: str = "cpu"
    num_rounds: int = 20
    client_fraction: float = 1.0
    min_clients_per_round: int = 1
    local_epochs: int = 1
    local_batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    max_grad_norm: Optional[float] = None

    num_clusters: int = 3
    partition_method: str = "kmeans"  # kmeans | label | random | single
    cluster_refresh_interval: int = 1
    max_partition_samples: int = 512
    min_examples_for_multimodal: int = 24

    mixture_conflict_beta: float = 1.0
    mixture_resolution: int = 4
    reference_sketch_mode: str = "unit"
    depth_weight_mode: str = "linear"

    lambda_r: float = 1.0
    lambda_c: float = 1.0

    eta_min: float = 0.10
    eta_max: float = 1.00
    mu_min: float = 0.00
    mu_max: float = 0.50
    alpha_credit: float = 4.0

    rho_min: float = 0.001
    rho_max: float = 0.050
    kappa_transfer: float = 6.0
    tau_transfer: float = 0.30
    probe_batch_size: int = 16

    aggregation: str = "weighted_mean"
    ensure_nonempty_gate: bool = True
    budget_scale: int = 200

    default_budget_fraction: float = 1.0
    default_threshold: float = -1e9

    controller: ControllerConfig = field(default_factory=ControllerConfig)
    ablations: AblationConfig = field(default_factory=AblationConfig)
