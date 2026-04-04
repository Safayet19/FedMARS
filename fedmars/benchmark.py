from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Subset, TensorDataset
from ucimlrepo import fetch_ucirepo

from .baselines import Ditto, FedAvg, FedDyn, FedOpt, FedProx, QFFL, SCAFFOLD
from .config import ControllerConfig, FedMARSConfig
from .core import FedMARS
from .data import ClientDataset, dirichlet_partition

UCI_DATASETS = {
    "sonar": {"id": 151},
    "digits": {"id": 80},
    "ckd": {"id": 336},
    "student_dropout": {"id": 697},
    "breast_cancer": {"id": 17},
}

METHODS = {
    "fedmars": FedMARS,
    "fedavg": FedAvg,
    "fedprox": FedProx,
    "scaffold": SCAFFOLD,
    "feddyn": FedDyn,
    "fedopt": FedOpt,
    "qffl": QFFL,
    "ditto": Ditto,
}


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip()
            out[col] = out[col].replace({"?": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})
    return out


def load_uci_dataset(dataset_name: str, seed: int = 42) -> dict[str, Any]:
    key = dataset_name.lower()
    if key not in UCI_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    data = fetch_ucirepo(id=UCI_DATASETS[key]["id"])
    X = data.data.features.copy()
    y = data.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            y = y.iloc[:, 0]
        else:
            y = y.astype(str).agg("__".join, axis=1)
    y = pd.Series(y).copy()
    X = _clean_dataframe(pd.DataFrame(X))
    y = y.astype(str).str.strip().replace({"?": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})
    mask = ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    X_train_df, X_temp_df, y_train_raw, y_temp_raw = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_val_df, X_test_df, y_val_raw, y_test_raw = train_test_split(X_temp_df, y_temp_raw, test_size=0.50, stratify=y_temp_raw, random_state=seed)
    numeric_cols = [col for col in X_train_df.columns if pd.api.types.is_numeric_dtype(X_train_df[col])]
    categorical_cols = [col for col in X_train_df.columns if col not in numeric_cols]
    transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols),
        ],
        remainder="drop",
    )
    X_train = transformer.fit_transform(X_train_df).astype(np.float32)
    X_val = transformer.transform(X_val_df).astype(np.float32)
    X_test = transformer.transform(X_test_df).astype(np.float32)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw.astype(str)).astype(np.int64)
    y_val = encoder.transform(y_val_raw.astype(str)).astype(np.int64)
    y_test = encoder.transform(y_test_raw.astype(str)).astype(np.int64)
    return {"name": key, "input_dim": int(X_train.shape[1]), "num_classes": int(len(encoder.classes_)), "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val, "X_test": X_test, "y_test": y_test}


def make_mlp(input_dim: int, num_classes: int) -> nn.Module:
    hidden1 = 128 if input_dim >= 32 else 64
    hidden2 = 64 if hidden1 >= 128 else 32
    return nn.Sequential(nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Linear(hidden2, num_classes))


def _split_client(client: ClientDataset, holdout_fraction: float, seed: int) -> tuple[ClientDataset, ClientDataset]:
    n = len(client.dataset)
    if n < 4:
        subset = Subset(client.dataset, list(range(n)))
        return client, ClientDataset(client.client_id, subset)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    holdout = max(1, int(round(n * holdout_fraction)))
    eval_idx = sorted(indices[:holdout].tolist())
    train_idx = sorted(indices[holdout:].tolist()) or eval_idx
    return ClientDataset(client.client_id, Subset(client.dataset, train_idx), metadata={**client.metadata, "split": "train"}), ClientDataset(client.client_id, Subset(client.dataset, eval_idx), metadata={**client.metadata, "split": "eval"})


def make_client_splits(X_train: np.ndarray, y_train: np.ndarray, num_clients: int = 10, alpha: float = 0.5, seed: int = 42, holdout_fraction: float = 0.20) -> tuple[list[ClientDataset], list[ClientDataset]]:
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    raw_clients = dirichlet_partition(dataset, num_clients=num_clients, alpha=alpha, seed=seed, min_size=max(8, len(X_train) // max(4 * num_clients, 1)))
    train_clients = []
    eval_clients = []
    for idx, client in enumerate(raw_clients):
        train_client, eval_client = _split_client(client, holdout_fraction, seed + idx)
        train_clients.append(train_client)
        eval_clients.append(eval_client)
    return train_clients, eval_clients


def default_config_for_dataset(dataset_name: str, seed: int = 42, device: str = "cpu") -> FedMARSConfig:
    small = dataset_name in {"sonar", "ckd", "breast_cancer"}
    return FedMARSConfig(
        random_state=seed,
        device=device,
        num_rounds=40 if dataset_name == "digits" else 30,
        warmup_rounds=5,
        local_epochs=2,
        local_batch_size=16 if small else 32,
        client_fraction=1.0,
        min_clients_per_round=5,
        num_clusters=2 if small else 3,
        num_batches_per_cluster=2 if small else 3,
        transfer_probe_batches=3,
        partition_method="kmeans" if dataset_name != "digits" else "label",
        mixture_conflict_beta=0.12,
        mixture_temperature=0.80,
        mixture_steps=50,
        lambda_r=0.18,
        lambda_c=0.01,
        lambda_v=0.20,
        probe_step=0.03,
        eta_min=0.60,
        eta_max=1.00,
        mu_min=0.00,
        mu_max=0.015,
        alpha_credit=1.20,
        rho_min=0.003,
        rho_max=0.035 if not small else 0.025,
        probe_batch_size=32 if small else 64,
        default_budget_fraction=1.00,
        default_threshold=-0.50,
        always_include_output_layer=True,
        nonselected_lr_scale=0.60,
        nonselected_mu_scale=1.10,
        freeze_unselected_after=999,
        aggregation_momentum=0.90,
        weight_decay=5e-5,
        controller=ControllerConfig(enabled=False),
    )


def _apply_overrides(config: FedMARSConfig, overrides: dict[str, Any] | None) -> FedMARSConfig:
    if not overrides:
        return config
    for key, value in overrides.items():
        if "." in key:
            head, tail = key.split(".", 1)
            setattr(getattr(config, head), tail, value)
        else:
            setattr(config, key, value)
    return config


def _instantiate(method: str, model: nn.Module, config: FedMARSConfig):
    key = method.lower()
    if key not in METHODS:
        raise ValueError(f"Unsupported method: {method}")
    return METHODS[key](model, config)


def summarize_history(method: str, history: dict[str, Any], client_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    last = history["rounds"][-1]
    validation = last.get("validation", {})
    test_metrics = history.get("test", {})
    summary = {
        "method": method,
        "round": int(last["round"]),
        "val_loss": float(validation.get("loss", 0.0)) if validation else 0.0,
        "val_acc": float(validation.get("accuracy", 0.0)) if validation else 0.0,
        "test_loss": float(test_metrics.get("loss", 0.0)) if test_metrics else 0.0,
        "test_acc": float(test_metrics.get("accuracy", 0.0)) if test_metrics else 0.0,
        "communication_ratio": float(last.get("communication_ratio", 1.0)),
        "client_to_server_bits": int(last.get("client_to_server_bits", 0)),
        "server_to_client_bits": int(last.get("server_to_client_bits", 0)),
        "total_bits": int(last.get("total_bits", 0)),
        "drift": float(last.get("drift", 0.0)),
        "selected_layers": list(last.get("selected_layers", [])),
    }
    if client_metrics is not None:
        summary.update({
            "client_mean_acc": float(client_metrics.get("mean_accuracy", 0.0)),
            "client_std_acc": float(client_metrics.get("std_accuracy", 0.0)),
            "client_worst_acc": float(client_metrics.get("worst_accuracy", 0.0)),
            "client_p10_acc": float(client_metrics.get("p10_accuracy", 0.0)),
        })
    return summary


def save_history_plot(history: dict[str, Any], filepath: str | Path, metric: str = "accuracy") -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    rounds = []
    vals = []
    bits = []
    total_bits = 0
    for row in history.get("rounds", []):
        rounds.append(int(row.get("round", len(rounds))))
        vals.append(float(row.get("validation", {}).get(metric, 0.0)))
        total_bits += int(row.get("client_to_server_bits", 0))
        bits.append(total_bits)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(rounds, vals, marker="o")
    plt.xlabel("round")
    plt.ylabel(metric)
    plt.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(bits, vals, marker="o")
    plt.xlabel("client_to_server_bits")
    plt.ylabel(metric)
    plt.tight_layout()
    fig.savefig(filepath.with_name(filepath.stem + "_bits" + filepath.suffix))
    plt.close(fig)


def run_experiment(method: str, dataset_name: str, output_dir: str | Path, seed: int = 42, alpha: float = 0.5, num_clients: int = 10, device: str = "cpu", config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    dataset = load_uci_dataset(dataset_name, seed)
    train_clients, eval_clients = make_client_splits(dataset["X_train"], dataset["y_train"], num_clients=num_clients, alpha=alpha, seed=seed)
    val_loader = DataLoader(TensorDataset(torch.tensor(dataset["X_val"], dtype=torch.float32), torch.tensor(dataset["y_val"], dtype=torch.long)), batch_size=256, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(dataset["X_test"], dtype=torch.float32), torch.tensor(dataset["y_test"], dtype=torch.long)), batch_size=256, shuffle=False)
    config = _apply_overrides(default_config_for_dataset(dataset_name, seed, device), config_overrides)
    trainer = _instantiate(method, make_mlp(dataset["input_dim"], dataset["num_classes"]), config)
    history = trainer.fit(train_clients, server_val_loader=val_loader, server_test_loader=test_loader)
    client_metrics = trainer.evaluate_personalized(eval_clients) if hasattr(trainer, "evaluate_personalized") else trainer.evaluate_clients(eval_clients)
    summary = summarize_history(method, history, client_metrics)
    payload = {"dataset": dataset_name, "method": method, "seed": seed, "alpha": alpha, "num_clients": num_clients, "config": asdict(config), "summary": _jsonable(summary), "client_metrics": _jsonable(client_metrics), "history": _jsonable(history)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    save_history_plot(history, output_dir / "validation_curve.png")
    return payload


def run_method_grid(dataset_name: str, methods: list[str], output_dir: str | Path, seed: int = 42, alpha: float = 0.5, num_clients: int = 10, device: str = "cpu", config_overrides: dict[str, Any] | None = None) -> pd.DataFrame:
    rows = []
    for method in methods:
        result = run_experiment(method, dataset_name, Path(output_dir) / method, seed, alpha, num_clients, device, config_overrides)
        rows.append(result["summary"])
    df = pd.DataFrame(rows).sort_values(by="test_acc", ascending=False).reset_index(drop=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(output_dir) / "summary.csv", index=False)
    return df
