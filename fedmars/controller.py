from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import ControllerConfig


@dataclass(frozen=True, slots=True)
class RoundState:
    drift: float
    communication_ratio: float
    validation_delta: float
    credit_mean: float


@dataclass(frozen=True, slots=True)
class ControllerAction:
    budget_fraction: float
    threshold: float
    action_index: int


class AdaptiveRoundController:
    def __init__(self, config: ControllerConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.actions = [
            (float(b), float(t))
            for b in config.budget_candidates
            for t in config.threshold_candidates
        ]
        self.q_table: Dict[tuple, np.ndarray] = {}

    def _bin_value(self, value: float, bins: tuple[float, ...]) -> int:
        for idx, edge in enumerate(bins):
            if value <= edge:
                return idx
        return len(bins)

    def state_key(self, state: RoundState) -> tuple[int, int, int, int]:
        return (
            self._bin_value(state.drift, self.config.drift_bins),
            self._bin_value(state.communication_ratio, self.config.comm_bins),
            self._bin_value(state.validation_delta, self.config.val_delta_bins),
            self._bin_value(state.credit_mean, self.config.credit_bins),
        )

    def choose(self, state: RoundState) -> ControllerAction:
        key = self.state_key(state)
        qvals = self.q_table.setdefault(key, np.zeros(len(self.actions), dtype=float))
        if self.rng.random() < self.config.epsilon:
            idx = int(self.rng.integers(0, len(self.actions)))
        else:
            idx = int(np.argmax(qvals))
        budget, threshold = self.actions[idx]
        return ControllerAction(budget_fraction=budget, threshold=threshold, action_index=idx)

    def update(self, state: RoundState, action: ControllerAction, reward: float) -> None:
        if action.action_index < 0:
            return
        key = self.state_key(state)
        qvals = self.q_table.setdefault(key, np.zeros(len(self.actions), dtype=float))
        qvals[action.action_index] += self.config.step_size * (reward - qvals[action.action_index])

    def compute_reward(self, validation_delta: float, communication_ratio: float, drift: float) -> float:
        return float(
            validation_delta
            - self.config.reward_comm_penalty * communication_ratio
            - self.config.reward_drift_penalty * drift
        )
