from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedmars.benchmark import run_experiment, run_method_grid

from pathlib import Path
import pandas as pd

DATASET = 'digits'
METHODS = ['fedmars', 'fedavg', 'fedprox', 'qffl', 'ditto']
OUT = Path('results/fairness_validation')
df = run_method_grid(DATASET, METHODS, OUT, seed=42, alpha=0.1, num_clients=10, device='cpu')
cols = ['method', 'test_acc', 'client_mean_acc', 'client_std_acc', 'client_worst_acc', 'client_p10_acc']
df[cols].to_csv(OUT / 'fairness_summary.csv', index=False)
print(df[cols].to_string(index=False))
