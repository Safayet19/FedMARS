# FedMARS

FedMARS is a research-oriented PyTorch package for a **credit-driven federated learning** workflow:
multimodal local gradients -> counterfactual mixture selection -> client layer credit -> global layer credit -> tri-control
(communication gate, server step size, proximal strength).

This package is designed as a **GitHub-installable reference implementation** for experiments, baselines, ablations, and external validation.

## Install from GitHub

```bash
pip install git+https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git
```

## Minimal usage

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from fedmars import ClientDataset, FedMARS, FedMARSConfig

model = nn.Sequential(
    nn.Linear(8, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)

clients = []
for cid in range(5):
    x = torch.randn(100, 8)
    y = ((x[:, 0] + 0.5 * x[:, 1] - 0.25 * x[:, 2]) > 0).long()
    clients.append(ClientDataset(client_id=cid, dataset=TensorDataset(x, y)))

config = FedMARSConfig(num_rounds=3, local_epochs=1, local_batch_size=16)
trainer = FedMARS(model=model, config=config)
result = trainer.fit(clients)
print(result["rounds"][-1]["selected_layers"])
```

## Package layout

- `fedmars/core.py`: main FedMARS algorithm.
- `fedmars/baselines.py`: FedAvg and FedProx reference baselines.
- `fedmars/config.py`: global configuration and ablation flags.
- `fedmars/controller.py`: adaptive round controller.
- `fedmars/credit.py`: reference sketch, layer credit, global credit.
- `fedmars/mixture.py`: counterfactual mixture selection.
- `fedmars/partition.py`: local multimodal partitioning.
- `fedmars/layers.py`: layer grouping and vector conversion.
- `fedmars/aggregation.py`: budgeted gating and update aggregation.
- `fedmars/data.py`: client dataset helpers and Dirichlet split utility.
- `fedmars/utils.py`: shared helper functions.

## Fixed main-design choices in this reference implementation

- linear depth weighting,
- residual conflict as the risk term,
- median aggregation for global layer credit,
- sigmoid mappings for server step size and proximal strength,
- small local cluster count,
- lightweight bandit-style round controller as a practical RL stand-in.
