import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedmars import ClientDataset, FedAvg, FedMARS, FedMARSConfig, FedProx


def make_model():
    return nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 2))


def make_clients():
    clients = []
    for cid in range(3):
        g = torch.Generator().manual_seed(123 + cid)
        x = torch.randn(64, 6, generator=g)
        y = ((x[:, 0] - 0.5 * x[:, 1] + 0.25 * x[:, 2]) > 0).long()
        clients.append(ClientDataset(client_id=cid, dataset=TensorDataset(x, y)))
    return clients


def test_fedmars_smoke():
    clients = make_clients()
    val_x = torch.randn(96, 6)
    val_y = ((val_x[:, 0] - 0.5 * val_x[:, 1] + 0.25 * val_x[:, 2]) > 0).long()
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32)
    cfg = FedMARSConfig(num_rounds=2, local_epochs=1, local_batch_size=16, num_clusters=2, client_fraction=1.0, partition_method="random")
    trainer = FedMARS(make_model(), cfg)
    history = trainer.fit(clients, server_val_loader=val_loader)
    assert len(history["rounds"]) == 2
    assert "selected_layers" in history["rounds"][0]


def test_baselines_smoke():
    clients = make_clients()
    val_x = torch.randn(96, 6)
    val_y = ((val_x[:, 0] - 0.5 * val_x[:, 1] + 0.25 * val_x[:, 2]) > 0).long()
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32)
    cfg = FedMARSConfig(num_rounds=1, local_epochs=1, local_batch_size=16, client_fraction=1.0)
    assert len(FedAvg(make_model(), cfg).fit(clients, server_val_loader=val_loader)["rounds"]) == 1
    assert len(FedProx(make_model(), cfg).fit(clients, server_val_loader=val_loader)["rounds"]) == 1
