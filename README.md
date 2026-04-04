# FedMARS

FedMARS is a PyTorch reference package for a unified layer-centric federated learning pipeline:

reference sketch -> repeated cluster mini-batches -> counterfactual mixture selection -> benefit-risk-cost-depth credit -> robust global credit -> tri-control -> transfer-based layer learning rate -> optional round controller.

## What is implemented

Core method:

- FedMARS

Reference baseline implementations included in this repo:

- FedAvg
- FedProx
- FedOpt
- SCAFFOLD
- FedDyn
- q-FFL
- Ditto

Dataset loaders included for the five requested UCI datasets:

- Sonar
- Optical Recognition of Handwritten Digits
- Chronic Kidney Disease
- Predict Students Dropout and Academic Success
- Breast Cancer Wisconsin Diagnostic

## Install

```bash
pip install -e .
```

## Quick check

```bash
pytest -q
python examples/quickstart_synthetic.py
python examples/compare_baselines.py
```

## Reproducible experiment entry points

- `experiments/run_single.py`: run one method on one dataset
- `experiments/run_grid.py`: run several methods on one dataset
- `experiments/sweep_heterogeneity.py`: Dirichlet alpha sweep
- `experiments/communication_validation.py`: communication-oriented evaluation
- `experiments/fairness_validation.py`: worst-client and variance evaluation
- `experiments/personalization_validation.py`: personalized evaluation
- `experiments/ablation_validation.py`: ablation suite
- `experiments/runners/`: 40 standardized dataset x method scripts
- `colab/`: Colab notebooks for training and validation workflows

## Important notes

- The UCI scripts require internet access at runtime because `ucimlrepo` downloads datasets from UCI.
- The round controller is optional and is **disabled by default** in the main config.
- The current repo gives a clean, standardized comparison framework for the implemented methods above. It does **not** claim to be an exact official implementation of every literature baseline named in the paper notes.

## Main package files

- `fedmars/core.py`: FedMARS training loop and round logging
- `fedmars/config.py`: configuration and ablation controls
- `fedmars/credit.py`: reference sketch and layer credit logic
- `fedmars/mixture.py`: counterfactual mixture selection
- `fedmars/aggregation.py`: budgeted layer selection and sparse aggregation
- `fedmars/baselines.py`: reference baseline implementations
- `fedmars/benchmark.py`: UCI dataset loading, preprocessing, experiment runners

## Logged fields

Each round log contains selected layers, communication ratio, client-to-server bits, server-to-client bits, total bits, drift, reward, layer credits, layer steps, proximal strengths, client transfer scores, and client layer learning rates.
