import torch
from torch import nn
from torch.utils.data import TensorDataset

from fedmars import ClientDataset, FedMARS, FedMARSConfig

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
clients = []
for cid in range(3):
    x = torch.randn(32, 4)
    y = ((x[:, 0] - 0.3 * x[:, 1]) > 0).long()
    clients.append(ClientDataset(client_id=cid, dataset=TensorDataset(x, y)))

trainer = FedMARS(model, FedMARSConfig(num_rounds=1, local_epochs=1, local_batch_size=8, num_clusters=2))
print(trainer.fit(clients)["rounds"][-1]["selected_layers"])
