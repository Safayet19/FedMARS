from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fedmars import FedAvg, FedMARS, FedMARSConfig, FedProx, dirichlet_partition


digits = load_digits()
X = digits.data.astype("float32") / 16.0
y = digits.target.astype("int64")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=256, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=256, shuffle=False)

clients = dirichlet_partition(train_dataset, num_clients=10, alpha=0.5, seed=42, min_size=16)


def make_model():
    return nn.Sequential(
        nn.Linear(X_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, len(set(y.tolist()))),
    )


config = FedMARSConfig(
    device="cpu",
    random_state=42,
    num_rounds=15,
    local_epochs=2,
    local_batch_size=32,
    client_fraction=1.0,
    min_clients_per_round=5,
    num_clusters=3,
    num_batches_per_cluster=3,
    partition_method="label",
    default_budget_fraction=0.70,
    default_threshold=-0.25,
)

fedmars = FedMARS(make_model(), config)
fedmars_history = fedmars.fit(clients, server_val_loader=val_loader, server_test_loader=test_loader)

fedavg = FedAvg(make_model(), config)
fedavg_history = fedavg.fit(clients, server_val_loader=val_loader, server_test_loader=test_loader)

fedprox = FedProx(make_model(), config)
fedprox_history = fedprox.fit(clients, server_val_loader=val_loader, server_test_loader=test_loader)

print("FedMARS validation:", fedmars_history["rounds"][-1].get("validation", {}))
print("FedMARS test:", fedmars_history.get("test", {}))
print("FedAvg validation:", fedavg_history["rounds"][-1].get("validation", {}))
print("FedAvg test:", fedavg_history.get("test", {}))
print("FedProx validation:", fedprox_history["rounds"][-1].get("validation", {}))
print("FedProx test:", fedprox_history.get("test", {}))
