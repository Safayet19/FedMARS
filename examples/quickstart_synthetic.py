import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedmars import ClientDataset, FedAvg, FedMARS, FedMARSConfig, FedProx


def make_client(seed: int, n: int = 120):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 8, generator=g)
    y = ((x[:, 0] + 0.7 * x[:, 1] - 0.5 * x[:, 2]) > 0).long()
    return ClientDataset(client_id=seed, dataset=TensorDataset(x, y))


def make_model():
    return nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))


clients = [make_client(seed) for seed in range(5)]
val_x = torch.randn(200, 8)
val_y = ((val_x[:, 0] + 0.7 * val_x[:, 1] - 0.5 * val_x[:, 2]) > 0).long()
val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=64)
config = FedMARSConfig(num_rounds=3, local_epochs=1, local_batch_size=16, client_fraction=1.0, min_clients_per_round=5)

fedmars = FedMARS(make_model(), config)
fedmars_history = fedmars.fit(clients, server_val_loader=val_loader)
print("FedMARS:", fedmars_history["rounds"][-1].get("validation", {}))

fedavg = FedAvg(make_model(), config)
fedavg_history = fedavg.fit(clients, server_val_loader=val_loader)
print("FedAvg:", fedavg_history["rounds"][-1].get("validation", {}))

fedprox = FedProx(make_model(), config)
fedprox_history = fedprox.fit(clients, server_val_loader=val_loader)
print("FedProx:", fedprox_history["rounds"][-1].get("validation", {}))
