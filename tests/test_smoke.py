import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedmars import FedAvg, FedMARS, FedMARSConfig, FedProx, ClientDataset, dirichlet_partition
from fedmars.aggregation import select_layers_under_budget
from fedmars.mixture import select_counterfactual_mixture


def make_model():
    return nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 2))


def make_clients():
    clients = []
    for cid in range(3):
        g = torch.Generator().manual_seed(123 + cid)
        x = torch.randn(24, 6, generator=g)
        y = ((x[:, 0] - 0.5 * x[:, 1] + 0.25 * x[:, 2]) > 0).long()
        clients.append(ClientDataset(client_id=cid, dataset=TensorDataset(x, y)))
    return clients


def test_select_layers_under_budget():
    credits = {"a": 1.0, "b": 0.5, "c": -0.1}
    costs = {"a": 0.6, "b": 0.3, "c": 0.1}
    chosen = select_layers_under_budget(credits, costs, budget_fraction=0.6, threshold=0.0)
    assert chosen == ["a"]


def test_counterfactual_mixture_simplex():
    grads = [torch.randn(10), torch.randn(10), torch.randn(10)]
    ref = torch.randn(10)
    weights, mixed, conflict, objective = select_counterfactual_mixture(grads, ref, beta=0.2, temperature=0.4, steps=20)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= -1e-6)
    assert mixed.shape == grads[0].shape
    assert isinstance(conflict, float)
    assert isinstance(objective, float)


def test_fedmars_smoke():
    clients = make_clients()
    val_x = torch.randn(48, 6)
    val_y = ((val_x[:, 0] - 0.5 * val_x[:, 1] + 0.25 * val_x[:, 2]) > 0).long()
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=16)
    cfg = FedMARSConfig(
        num_rounds=1,
        local_epochs=1,
        local_batch_size=8,
        client_fraction=1.0,
        min_clients_per_round=3,
        num_clusters=2,
        num_batches_per_cluster=2,
        partition_method="random",
        default_budget_fraction=0.75,
        default_threshold=-0.5,
        probe_batch_size=16,
        transfer_probe_batches=2,
    )
    trainer = FedMARS(make_model(), cfg)
    history = trainer.fit(clients, server_val_loader=val_loader)
    assert len(history["rounds"]) == 1
    assert "selected_layers" in history["rounds"][0]
    assert history["rounds"][0]["selected_layers"]
    assert "client_to_server_bits" in history["rounds"][0]
    assert "client_layer_lrs" in history["rounds"][0]


def test_baselines_smoke():
    clients = make_clients()
    val_x = torch.randn(48, 6)
    val_y = ((val_x[:, 0] - 0.5 * val_x[:, 1] + 0.25 * val_x[:, 2]) > 0).long()
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=16)
    cfg = FedMARSConfig(num_rounds=1, local_epochs=1, local_batch_size=8, client_fraction=1.0, min_clients_per_round=3)
    assert len(FedAvg(make_model(), cfg).fit(clients, server_val_loader=val_loader)["rounds"]) == 1
    assert len(FedProx(make_model(), cfg).fit(clients, server_val_loader=val_loader)["rounds"]) == 1


def test_dirichlet_partition():
    g = torch.Generator().manual_seed(77)
    x = torch.randn(96, 5, generator=g)
    y = torch.randint(0, 4, (96,), generator=g)
    dataset = TensorDataset(x, y)
    clients = dirichlet_partition(dataset, num_clients=6, alpha=0.5, seed=42, min_size=6)
    assert len(clients) == 6
    assert min(len(c) for c in clients) >= 6
