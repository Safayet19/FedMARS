from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedmars.benchmark import run_experiment, run_method_grid

from pathlib import Path
import pandas as pd

DATASET = 'digits'
METHODS = ['fedmars', 'fedavg', 'fedprox', 'scaffold', 'feddyn', 'fedopt']
ALPHAS = [1.0, 0.5, 0.1, 0.05, 0.01]
OUT = Path('results/heterogeneity_sweep')
rows = []
for alpha in ALPHAS:
    df = run_method_grid(DATASET, METHODS, OUT / f'alpha_{alpha}', seed=42, alpha=alpha, num_clients=10, device='cpu')
    df['alpha'] = alpha
    rows.append(df)
pd.concat(rows, ignore_index=True).to_csv(OUT / 'summary.csv', index=False)
print((OUT / 'summary.csv').as_posix())
