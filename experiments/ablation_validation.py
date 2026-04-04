from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedmars.benchmark import run_experiment, run_method_grid

from pathlib import Path
import pandas as pd

from fedmars.benchmark import run_experiment

DATASET = 'digits'
OUT = Path('results/ablation_validation')
ablations = {
    'full': {},
    'no_mixture': {'ablations.use_counterfactual_mixture': False},
    'no_credit': {'ablations.use_layer_credit': False},
    'no_transfer_lr': {'ablations.use_transfer_lr': False},
    'no_round_controller': {'ablations.use_round_controller': False},
    'no_train_gate': {'ablations.use_train_gate': False},
}
rows = []
for name, overrides in ablations.items():
    result = run_experiment('fedmars', DATASET, OUT / name, seed=42, alpha=0.1, num_clients=10, device='cpu', config_overrides=overrides)
    row = result['summary']
    row['ablation'] = name
    rows.append(row)
pd.DataFrame(rows).sort_values('test_acc', ascending=False).to_csv(OUT / 'ablation_summary.csv', index=False)
print((OUT / 'ablation_summary.csv').as_posix())
