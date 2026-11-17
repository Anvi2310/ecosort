"""Bounded Gaussian membership functions and a compact three-set partition."""
from dataclasses import dataclass
import numpy as np


@dataclass
class GaussianMF:
    center: float
    sigma: float

    def __post_init__(self):
        self.center = float(self.center)
        self.sigma = float(self.sigma)
        if not 0 <= self.center <= 100 or not 1 <= self.sigma <= 60:
            raise ValueError("Gaussian parameters out of bounds")

    def __call__(self, value):
        x = np.asarray(value, dtype=float)
        return np.exp(-0.5 * ((x - self.center) / self.sigma) ** 2)


def default_partition():
    return {"low": GaussianMF(15, 18), "medium": GaussianMF(50, 18), "high": GaussianMF(85, 18)}


def degree(value, label, partition=None):
    p = partition or default_partition()
    try:
        return float(p[label.lower()](value))
    except KeyError as exc:
        raise ValueError(f"Unknown membership label: {label}") from exc


def partition_payload(partition=None, points=101):
    p = partition or default_partition()
    xs = np.linspace(0, 100, points)
    return {name: {"center": mf.center, "sigma": mf.sigma,
                   "x": xs.tolist(), "y": mf(xs).tolist()} for name, mf in p.items()}
