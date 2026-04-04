from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedmars.benchmark import run_experiment, run_method_grid

from pathlib import Path
import pandas as pd

DATASET = 'digits'
METHODS = ['fedmars', 'fedavg', 'fedopt']
BUDGETS = [1.0, 0.7, 0.5, 0.3]
OUT = Path('results/communication_validation')
rows = []
for budget in BUDGETS:
    overrides = {'default_budget_fraction': budget}
    df = run_method_grid(DATASET, METHODS, OUT / f'budget_{budget}', seed=42, alpha=0.5, num_clients=10, device='cpu', config_overrides=overrides)
    df['budget_fraction'] = budget
    rows.append(df)
pd.concat(rows, ignore_index=True).to_csv(OUT / 'summary.csv', index=False)
print((OUT / 'summary.csv').as_posix())
