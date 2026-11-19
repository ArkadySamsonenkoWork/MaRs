import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, root_dir)

import torch
import json
import typing as tp
from pathlib import Path
import datetime

import mars


def save_benchmark_results(results: tp.Dict, filepath: tp.Union[str, Path]) -> None:
    """
    Save benchmark results dictionary to JSON file with metadata.

    Parameters
    ----------
    results : dict
        Dictionary returned by benchmark_all_configurations()
    filepath : str or Path
        Output file path (e.g., "benchmarks/run_20260203.json")

    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    full_data = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "mars_version": getattr(mars, "__version__", "unknown"),
            "pytorch_version": torch.__version__,
            "device": str(next(iter(results.values())).get("device", "unknown")) if results else "unknown",
            "dtype": str(next(iter(results.values())).get("dtype", "unknown")) if results else "unknown",
            "total_configurations": sum(len(mesh_data) for mesh_data in results.values())
        },
        "results": results
    }

    with open(filepath, 'w') as f:
        json.dump(full_data, f, indent=2, default=str)

    print(f"✓ Saved benchmark results to {filepath}")


def load_benchmark_results(filepath: tp.Union[str, Path]) -> tp.Dict:
    """
    Load benchmark results from JSON file.

    Parameters
    ----------
    filepath : str or Path
        Path to saved benchmark JSON file

    Returns
    -------
    results : dict
        Full results dictionary including metadata and timing data

    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Benchmark file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    print(f"✓ Loaded benchmark from {filepath.name}")
    print(f"  Timestamp: {data['metadata']['timestamp']}")
    print(f"  Mars v{data['metadata']['mars_version']} | PyTorch v{data['metadata']['pytorch_version']}")
    print(f"  Configurations: {data['metadata']['total_configurations']}")

    return data