from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedmars.benchmark import run_experiment, run_method_grid

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--methods', nargs='+', required=True)
parser.add_argument('--output_dir', default='results/grid_run')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--num_clients', type=int, default=10)
parser.add_argument('--device', default='cpu')
args = parser.parse_args()

df = run_method_grid(args.dataset, args.methods, args.output_dir, seed=args.seed, alpha=args.alpha, num_clients=args.num_clients, device=args.device)
print(df.to_string(index=False))
