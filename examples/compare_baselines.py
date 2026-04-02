"""Comparison template.

Keep the same:
- model architecture
- client split
- round count
- local epochs
- validation and test loaders

Then run FedAvg, FedProx, and FedMARS on exactly that shared setup.
"""
