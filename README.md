# FedMARS

FedMARS is a PyTorch reference package for a unified layer-centric federated learning pipeline:

global direction reference -> multi-cluster multi-mini-batch gradients -> counterfactual mixture selection -> benefit-risk-cost-depth credit -> robust global credit aggregation -> tri-control -> transfer-based layer learning rate -> adaptive round controller.

## Install

```bash
pip install -e .
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
history = trainer.fit(clients)
print(history["rounds"][-1]["selected_layers"])
```

## Main package files

- `fedmars/core.py`: full FedMARS training loop and round logging.
- `fedmars/config.py`: all configuration knobs, controller config, ablation config.
- `fedmars/credit.py`: reference sketch, layer credit, global credit aggregation, control-credit postprocessing.
- `fedmars/mixture.py`: counterfactual mixture optimization and conflict computation.
- `fedmars/partition.py`: local clustering and repeated mini-batch sampling.
- `fedmars/layers.py`: layer grouping, depth weights, costs, bits, vector utilities.
- `fedmars/aggregation.py`: budgeted layer selection, sparse aggregation, server momentum, global update.
- `fedmars/controller.py`: adaptive round controller.
- `fedmars/baselines.py`: FedAvg and FedProx reference baselines.
- `fedmars/data.py`: client dataset helpers and Dirichlet partitioning.
- `fedmars/utils.py`: shared utilities.
- `examples/`: runnable usage examples.
- `tests/`: smoke tests for the package.

## Default main-design choices in this implementation

- linear depth weighting,
- repeated mini-batch gradients inside each client cluster,
- counterfactual mixture selection on cluster gradients,
- residual conflict as the risk term,
- median global credit aggregation,
- robust-zscore control credit for stable tri-control mapping,
- transferability from repeated probe-batch gradient agreement,
- optional adaptive round controller enabled by default,
- clipped weighted aggregation for sparse server updates.

## Logged fields

Each round log contains:

- sampled clients,
- controller action,
- selected layers,
- raw global credit,
- control credit used for tri-control,
- layer-wise server step sizes,
- layer-wise proximal strengths,
- communication ratio,
- client-to-server bits,
- server-to-client bits,
- total bits,
- drift,
- reward,
- client credit diagnostics,
- transfer scores,
- layer learning rates.

## Examples

Run a quick synthetic smoke example:

```bash
python examples/quickstart_synthetic.py
```

Run an offline digits comparison:

```bash
python examples/compare_baselines.py
```
