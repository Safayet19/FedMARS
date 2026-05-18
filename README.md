# FedMARS

FedMARS is a PyTorch implementation of Federated Mode-Aware Reliability Scoring for layer-wise trust control in heterogeneous federated learning. The method builds mode-aware layer evidence on each client, converts that evidence into a layer credit, robustly aggregates credits on the server, and uses the same credit for layer selection, server update strength, and proximal drift control.

## Install

```bash
pip install .
```

For the matching package release:

```bash
pip install fedmars==0.5.1
```

## Minimal use

```python
from fedmars import FedMARS, FedMARSConfig

config = FedMARSConfig(num_rounds=40, default_budget_fraction=0.35)
trainer = FedMARS(model, config)
history = trainer.fit(clients, server_val_loader=val_loader, server_test_loader=test_loader)
```

Each client must be a `fedmars.ClientDataset` or compatible object with `client_id`, `dataset`, and optional `weight` fields. Datasets must return `(x, y)` pairs.

## Implementation choices in this submission version

| Area | Submission behavior |
|---|---|
| Layer budget | Exact parameter-bit budget: selected layer bits are constrained by `floor(model_bits * budget_fraction)`. |
| Output layer | No layer is forced by default. If no layer fits a very small budget, the selected set can be empty without violating the bit budget. |
| No-reference credit | Final credit alignment is neutral when no usable server reference exists. |
| Mixture objective | Entropy is controlled by the explicit `mixture_entropy` field. |
| Server momentum | EMA update: `v <- m v + (1-m) delta`. |
| Aggregation weighting | Positive credit weighting uses robust z-normalized client-layer credit. |
| Communication logs | Uplink, downlink, total bits, selected bits, budget bits, model bits, and budget-violation flags are recorded each round. |

The source package contains only the FedMARS architecture, data/layer utilities, and method-critical unit tests. Experiment notebooks, smoke-result CSVs, and ablation-study scripts are intentionally excluded from this submission code package.
