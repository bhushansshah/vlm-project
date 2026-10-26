"""Benchmark registry. Add a dataset by subclassing ``Dataset`` and listing it here."""

from __future__ import annotations

from pathlib import Path

from .base import Dataset, Sample, Selection, select_samples
from .treebench import TreeBench
from .vistaqa import VistaQA
from .visualoverload import VisualOverload

DATASETS: dict[str, type[Dataset]] = {d.name: d for d in (VisualOverload, VistaQA, TreeBench)}


def get_dataset(name: str, data_dir: str | Path | None = None) -> Dataset:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {sorted(DATASETS)}")
    return DATASETS[name](data_dir)


__all__ = ["DATASETS", "Dataset", "Sample", "Selection", "get_dataset", "select_samples"]
