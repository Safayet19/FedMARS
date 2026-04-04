from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedmars.benchmark import run_experiment, run_method_grid

from pathlib import Path
import pandas as pd

DATASETS = ['sonar', 'digits', 'ckd', 'student_dropout', 'breast_cancer']
METHODS = ['fedmars', 'fedavg', 'fedprox', 'scaffold', 'feddyn', 'fedopt', 'qffl', 'ditto']
OUT = Path('results/full_validation_suite')
frames = []
for dataset in DATASETS:
    df = run_method_grid(dataset, METHODS, OUT / dataset, seed=42, alpha=0.5, num_clients=10, device='cpu')
    df['dataset'] = dataset
    frames.append(df)
pd.concat(frames, ignore_index=True).to_csv(OUT / 'summary.csv', index=False)
print((OUT / 'summary.csv').as_posix())
