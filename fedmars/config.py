from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(slots=True)
class ControllerConfig:
    enabled: bool = True
    budget_candidates: Tuple[float, ...] = (0.35, 0.50, 0.70, 1.00)
    threshold_candidates: Tuple[float, ...] = (-0.50, -0.25, 0.00, 0.25)
    epsilon: float = 0.12
    step_size: float = 0.25
    drift_bins: Tuple[float, ...] = (0.002, 0.01, 0.05, 0.20)
    comm_bins: Tuple[float, ...] = (0.20, 0.40, 0.60, 0.85)
    val_delta_bins: Tuple[float, ...] = (-0.02, -0.005, 0.005, 0.02)
    credit_bins: Tuple[float, ...] = (-1.00, -0.25, 0.25, 1.00)
    reward_comm_penalty: float = 0.08
    reward_drift_penalty: float = 0.03


@dataclass(slots=True)
class AblationConfig:
    use_reference_sketch: bool = True
    use_multimodal_partition: bool = True
    use_counterfactual_mixture: bool = True
    use_layer_credit: bool = True
    use_transfer_lr: bool = True
    use_round_controller: bool = True
    use_depth_weight: bool = True
    use_train_gate: bool = False
    use_credit_weighted_aggregation: bool = True


@dataclass(slots=True)
class FedMARSConfig:
    random_state: int = 42
    device: str = "cpu"

    num_rounds: int = 30
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
    partition_method: str = "label"
    cluster_refresh_interval: int = 5
    max_partition_samples: int = 512
    min_examples_for_multimodal: int = 24

    mixture_conflict_beta: float = 0.35
    mixture_temperature: float = 0.40
    mixture_steps: int = 40
    mixture_resolution: int = 4

    reference_momentum: float = 0.80
    reference_sketch_mode: str = "ema_unit"
    reference_topk_fraction: float = 0.10
    depth_weight_mode: str = "linear"

    lambda_r: float = 0.80
    lambda_c: float = 0.10
    lambda_v: float = 0.00
    probe_step: float = 0.05

    eta_min: float = 0.20
    eta_max: float = 0.90
    mu_min: float = 0.00
    mu_max: float = 0.08
    alpha_credit: float = 2.50

    rho_min: float = 0.002
    rho_max: float = 0.030
    kappa_transfer: float = 3.0
    tau_transfer: float = 0.10
    probe_batch_size: int = 64
    transfer_probe_batches: int = 2

    global_credit_aggregator: str = "median"
    control_credit_mode: str = "robust_zscore"
    control_credit_clip: float = 2.50

    aggregation: str = "clipped_weighted_mean"
    ensure_nonempty_gate: bool = True
    budget_scale: int = 200
    default_budget_fraction: float = 0.70
    default_threshold: float = -0.25

    nonselected_lr_scale: float = 0.50
    nonselected_mu_scale: float = 1.50
    freeze_unselected_after: int = 999999

    aggregation_momentum: float = 0.00
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0

    param_bits: int = 32
    track_server_to_client_bits: bool = True

    controller: ControllerConfig = field(default_factory=ControllerConfig)
    ablations: AblationConfig = field(default_factory=AblationConfig)

    def __post_init__(self) -> None:
        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be positive.")
        if self.warmup_rounds < 0 or self.positive_pair_rounds < 0:
            raise ValueError("warmup_rounds and positive_pair_rounds must be non-negative.")
        if not (0.0 < self.client_fraction <= 1.0):
            raise ValueError("client_fraction must be in (0, 1].")
        if self.min_clients_per_round <= 0:
            raise ValueError("min_clients_per_round must be positive.")
        if self.local_epochs <= 0:
            raise ValueError("local_epochs must be positive.")
        if self.local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive.")
        if self.num_clusters <= 0:
            raise ValueError("num_clusters must be positive.")
        if self.num_batches_per_cluster <= 0:
            raise ValueError("num_batches_per_cluster must be positive.")
        if self.cluster_refresh_interval <= 0:
            raise ValueError("cluster_refresh_interval must be positive.")
        if self.max_partition_samples <= 0:
            raise ValueError("max_partition_samples must be positive.")
        if self.min_examples_for_multimodal <= 0:
            raise ValueError("min_examples_for_multimodal must be positive.")
        if self.mixture_steps <= 0:
            raise ValueError("mixture_steps must be positive.")
        if not (0.0 <= self.default_budget_fraction <= 1.0):
            raise ValueError("default_budget_fraction must be in [0, 1].")
        if not (0.0 <= self.reference_topk_fraction <= 1.0):
            raise ValueError("reference_topk_fraction must be in [0, 1].")
        if self.rho_min <= 0.0 or self.rho_max <= 0.0 or self.rho_max < self.rho_min:
            raise ValueError("rho_min and rho_max must be positive with rho_max >= rho_min.")
        if self.eta_min < 0.0 or self.eta_max < self.eta_min:
            raise ValueError("eta_min and eta_max must satisfy 0 <= eta_min <= eta_max.")
        if self.mu_min < 0.0 or self.mu_max < self.mu_min:
            raise ValueError("mu_min and mu_max must satisfy 0 <= mu_min <= mu_max.")
        if self.param_bits <= 0:
            raise ValueError("param_bits must be positive.")
        if self.global_credit_aggregator not in {"median", "trimmed_mean", "clipped_mean"}:
            raise ValueError("Unsupported global_credit_aggregator.")
        if self.control_credit_mode not in {"none", "robust_zscore"}:
            raise ValueError("Unsupported control_credit_mode.")
        if self.aggregation not in {"weighted_mean", "clipped_weighted_mean"}:
            raise ValueError("Unsupported aggregation.")
