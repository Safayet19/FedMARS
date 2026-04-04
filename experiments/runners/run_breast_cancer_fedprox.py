from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedmars.benchmark import run_experiment, run_method_grid

result = run_experiment('fedprox', 'breast_cancer', 'results/breast_cancer/fedprox', seed=42, alpha=0.5, num_clients=10, device='cpu')
print(result['summary'])
