import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, root_dir)

import time
import torch
import typing as tp
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from mars import spin_model, mesher, constants, spectra_manager, population


def time_spectrum_calculation(
        sample: spin_model.SolidSample,
        freq: float = 9.8e9,
        field_range: tp.Tuple[float, float] = (0.30, 0.40),
        n_points: int = 1000,
        n_warmup: int = 5,
        n_iterations: int = 50,
        temperature: float = 298.0,
        computational_details: spectra_manager.ComputationalDetails = spectra_manager.ComputationalDetails(),
) -> tp.Tuple[tp.Union[float, np.ndarray], tp.Union[float, np.ndarray], tp.List[float]]:
    """
    Measure spectrum calculation time with warmup iterations.


    Parameters
    ----------
    sample : SolidSample
        Pre-configured spin system sample.
    freq : float, optional
        Microwave frequency in Hz. Default is 9.8 GHz (X-band).
    field_range : tuple of (float, float), optional
        Magnetic field range (min, max) in Tesla.
    n_points : int, optional
        Number of field points in simulation. Default is 1000.
    n_warmup : int, optional
        Number of warmup iterations (discarded from timing). Default is 2.
    n_iterations : int, optional
        Number of timed iterations. Default is 5.
    temperature : float, optional
        Sample temperature in Kelvin. Default is 298 K (room temp).

    Returns
    -------
    mean_time_ms : float
        Mean execution time in milliseconds.
    std_time_ms : float
        Standard deviation of execution times in milliseconds.
    all_times_ms : list of float
        Raw timing measurements for all iterations.
    """
    device = sample.device
    dtype = sample.dtype

    fields = torch.linspace(
        field_range[0],
        field_range[1],
        n_points,
        device=device,
        dtype=dtype
    )

    creator = spectra_manager.StationarySpectra(
        freq=freq,
        sample=sample,
        temperature=temperature,
        computational_details=computational_details,
        device=device,
        dtype=dtype
    )

    for _ in range(n_warmup):
        _ = creator(sample, fields)

    times_ms = []
    for _ in range(n_iterations):
        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.perf_counter()

        _ = creator(sample, fields)

        torch.cuda.synchronize() if device.type == "cuda" else None
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000.0  # Convert to milliseconds
        times_ms.append(elapsed_ms)

    mean_time = np.mean(times_ms)
    std_time = np.std(times_ms)

    return mean_time, std_time, times_ms


def _prepare_x_axis(
        field_range: tp.Tuple[float, float],
        n_points: int,
        device: torch.device,
        dtype: torch.dtype,
        sample: "spin_model.SolidSample"
) -> torch.Tensor:
    """
    Create field tensor, handling batch expansion if necessary.

    Checks sample.config_shape to determine if data is batched.
    If len(config_shape) == 2, expands fields to [batch_size, n_points].
    """
    fields = torch.linspace(
        field_range[0],
        field_range[1],
        n_points,
        device=device,
        dtype=dtype
    )

    if len(sample.config_shape) == 2:
        batch_size = sample.config_shape[0]
        fields = fields.expand(batch_size, -1)

    return fields


def time_spectrum_calculation_full_pipeline(
        sample_creation_func: tp.Callable[..., spin_model.SolidSample],
        context: tp.Optional[tp.Union[population.Context, tp.Callable]] = None,
        sample_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        creator_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        freq: float = 9.8e9,
        field_range: tp.Tuple[float, float] = (0.30, 0.40),
        n_points: int = 1000,
        n_warmup: int = 5,
        n_iterations: int = 50,
        temperature: float = 298.0,
        computational_details: spectra_manager.ComputationalDetails = spectra_manager.ComputationalDetails(),
) -> tp.Tuple[tp.Union[float, np.ndarray], tp.Union[float, np.ndarray], tp.List[float]]:
    """
    Measure full spectrum calculation time including sample creation.

    This function measures the complete execution time including:
    - Sample creation (spin system, interactions, etc.)
    - StationarySpectra object initialization
    - Spectrum computation
    - All associated overhead

    Parameters
    ----------
    sample_creation_func : callable
        Function that creates a SolidSample (e.g., create_2_electrons_sample).
    sample_kwargs : dict, optional
        Keyword arguments to pass to the sample creation function.
        Default is None (uses function defaults).
    freq : float, optional
        Microwave frequency in Hz. Default is 9.8 GHz (X-band).
    field_range : tuple of (float, float), optional
        Magnetic field range (min, max) in Tesla.
    n_points : int, optional
        Number of field points in simulation. Default is 1000.
    n_warmup : int, optional
        Number of warmup iterations (discarded from timing). Default is 5.
    n_iterations : int, optional
        Number of timed iterations. Default is 50.
    temperature : float, optional
        Sample temperature in Kelvin. Default is 298 K (room temp).

    Returns
    -------
    mean_time_ms : float
        Mean execution time in milliseconds (full pipeline).
    std_time_ms : float
        Standard deviation of execution times in milliseconds.
    all_times_ms : list of float
        Raw timing measurements for all iterations.

    See Also
    --------
    time_spectrum_calculation : Measures only computation time (excludes init).
    time_spectrum_calculation_with_init : Measures computation + spectra init (excludes sample creation).

    """
    if sample_kwargs is None:
        sample_kwargs = {}

    if creator_kwargs is None:
        creator_kwargs = {}

    is_context_callable = callable(context)

    if context is None:
        creator_kwargs["harmonic"] = 1
        creator_kwargs.pop("context", None)
    elif is_context_callable:
        creator_kwargs["harmonic"] = 0
        creator_kwargs.pop("context", None)  # Will be added inside the loop
    else:
        creator_kwargs["harmonic"] = 0
        creator_kwargs["context"] = context

    times_ms = []
    for _ in range(n_warmup):
        sample = sample_creation_func(**sample_kwargs)
        device = sample.device
        dtype = sample.dtype

        if is_context_callable:
            creator_kwargs["context"] = context(sample)

        fields = _prepare_x_axis(field_range, n_points, device, dtype, sample)

        creator = spectra_manager.StationarySpectra(
            freq=freq,
            sample=sample,
            temperature=temperature,
            computational_details=computational_details,
            device=device,
            dtype=dtype,
            **creator_kwargs,
        )
        _ = creator(sample, fields)
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(n_iterations):
        if sample_kwargs.get("device", torch.device("cpu")).type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        sample = sample_creation_func(**sample_kwargs)
        device = sample.device
        dtype = sample.dtype

        fields = _prepare_x_axis(field_range, n_points, device, dtype, sample)
        creator = spectra_manager.StationarySpectra(
            freq=freq,
            sample=sample,
            temperature=temperature,
            computational_details=computational_details,
            device=device,
            dtype=dtype,
            **creator_kwargs
        )
        _ = creator(sample, fields)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000.0
        times_ms.append(elapsed_ms)

    mean_time = np.mean(times_ms)
    std_time = np.std(times_ms)

    return mean_time, std_time, times_ms


def time_spectrum_calculation_full_pipeline_freqdep(
        sample_creation_func: tp.Callable[..., spin_model.SolidSample],
        context: tp.Optional[tp.Union[population.Context, tp.Callable]] = None,
        sample_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        creator_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        field: float = 10.0,
        freq_range: tp.Tuple[float, float] = (0.0, 1e9),
        n_points: int = 1000,
        n_warmup: int = 5,
        n_iterations: int = 50,
        temperature: float = 298.0,
        computational_details: spectra_manager.ComputationalDetails = spectra_manager.ComputationalDetails(),
) -> tp.Tuple[tp.Union[float, np.ndarray], tp.Union[float, np.ndarray], tp.List[float]]:
    """
    Measure full spectrum calculation time including sample creation.

    This function measures the complete execution time including:
    - Sample creation (spin system, interactions, etc.)
    - StationaryFreq object initialization
    - Spectrum computation
    - All associated overhead

    Parameters
    ----------
    sample_creation_func : callable
        Function that creates a SolidSample (e.g., create_2_electrons_sample).
    sample_kwargs : dict, optional
        Keyword arguments to pass to the sample creation function.
        Default is None (uses function defaults).
    field : float, optional
        Stationary magnetic field in Hz. Default is 9.8 GHz (X-band).
    freq_range : tuple of (float, float), optional
        Microwave freq range (min, max) in Tesla.
    n_points : int, optional
        Number of field points in simulation. Default is 1000.
    n_warmup : int, optional
        Number of warmup iterations (discarded from timing). Default is 5.
    n_iterations : int, optional
        Number of timed iterations. Default is 50.
    temperature : float, optional
        Sample temperature in Kelvin. Default is 298 K (room temp).

    Returns
    -------
    mean_time_ms : float
        Mean execution time in milliseconds (full pipeline).
    std_time_ms : float
        Standard deviation of execution times in milliseconds.
    all_times_ms : list of float
        Raw timing measurements for all iterations.

    See Also
    --------
    time_spectrum_calculation : Measures only computation time (excludes init).
    time_spectrum_calculation_with_init : Measures computation + spectra init (excludes sample creation).

    """
    if sample_kwargs is None:
        sample_kwargs = {}

    if creator_kwargs is None:
        creator_kwargs = {}

    if context is None:
        creator_kwargs["harmonic"] = 0
    else:
        creator_kwargs["context"] = context
        creator_kwargs["harmonic"] = 0

    times_ms = []
    for _ in range(n_warmup):
        sample = sample_creation_func(**sample_kwargs)
        device = sample.device
        dtype = sample.dtype

        freq = torch.linspace(
            freq_range[0],
            freq_range[1],
            n_points,
            device=device,
            dtype=dtype
        )

        creator = spectra_manager.StationaryFreqSpectra(
            field=field,
            sample=sample,
            temperature=temperature,
            computational_details=computational_details,
            device=device,
            dtype=dtype,
            **creator_kwargs,
        )
        _ = creator(sample, freq)
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(n_iterations):
        if sample_kwargs.get("device", torch.device("cpu")).type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        sample = sample_creation_func(**sample_kwargs)
        device = sample.device
        dtype = sample.dtype

        freq = torch.linspace(
            freq_range[0],
            freq_range[1],
            n_points,
            device=device,
            dtype=dtype
        )
        creator = spectra_manager.StationaryFreqSpectra(
            field=field,
            sample=sample,
            temperature=temperature,
            computational_details=computational_details,
            device=device,
            dtype=dtype,
            **creator_kwargs
        )
        _ = creator(sample, freq)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000.0
        times_ms.append(elapsed_ms)

    mean_time = np.mean(times_ms)
    std_time = np.std(times_ms)

    return mean_time, std_time, times_ms


def time_spectrum_calculation_full_pipeline_timedep(
        sample_creation_func: tp.Callable[..., spin_model.SolidSample],
        context: tp.Optional[tp.Union[population.Context, tp.Callable]] = None,
        sample_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        creator_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        freq: float = 9.8e9,
        field_range: tp.Tuple[float, float] = (0.30, 0.40),
        time_range: tp.Tuple[float, float] = (0.0, 1e-4),
        n_points: int = 1000,
        n_points_time: int = 100,
        n_warmup: int = 5,
        n_iterations: int = 50,
        temperature: float = 298.0,
        simulation_cls=spectra_manager.CoupledTimeSpectra,
        computational_details: spectra_manager.ComputationalDetails = spectra_manager.ComputationalDetails(),
) -> tp.Tuple[tp.Union[float, np.ndarray], tp.Union[float, np.ndarray], tp.List[float]]:
    """
    """
    if sample_kwargs is None:
        sample_kwargs = {}

    if creator_kwargs is None:
        creator_kwargs = {}

    is_context_callable = callable(context)

    if context is None:
        creator_kwargs["harmonic"] = 1
        creator_kwargs.pop("context", None)
    elif is_context_callable:
        creator_kwargs["harmonic"] = 0
        creator_kwargs.pop("context", None)  # Will be added inside the loop
    else:
        creator_kwargs["harmonic"] = 0
        creator_kwargs["context"] = context

    times_ms = []
    for _ in range(n_warmup):
        sample = sample_creation_func(**sample_kwargs)
        device = sample.device
        dtype = sample.dtype

        if is_context_callable:
            creator_kwargs["context"] = context(sample)

        fields = _prepare_x_axis(field_range, n_points, device, dtype, sample)
        time_simulation = _prepare_x_axis(time_range, n_points_time, device, dtype, sample)

        creator = simulation_cls(
            freq=freq,
            sample=sample,
            temperature=temperature,
            computational_details=computational_details,
            device=device,
            dtype=dtype,
            **creator_kwargs,
        )
        _ = creator(sample, fields, time_simulation)
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(n_iterations):
        if sample_kwargs.get("device", torch.device("cpu")).type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        sample = sample_creation_func(**sample_kwargs)
        device = sample.device
        dtype = sample.dtype

        fields = _prepare_x_axis(field_range, n_points, device, dtype, sample)
        time_simulation = _prepare_x_axis(time_range, n_points_time, device, dtype, sample)

        creator = simulation_cls(
            freq=freq,
            sample=sample,
            temperature=temperature,
            computational_details=computational_details,
            device=device,
            dtype=dtype,
            **creator_kwargs
        )
        _ = creator(sample, fields, time_simulation)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000.0
        times_ms.append(elapsed_ms)

    mean_time = np.mean(times_ms)
    std_time = np.std(times_ms)

    return mean_time, std_time, times_ms


def _plot_benchmark_results(results: dict, device: torch.device, dtype: torch.dtype) -> None:
    """Helper function to visualize benchmark results with mesh size annotations and system labels."""
    plt.figure(figsize=(14, 8))

    color_map = {
        "2e": "#1f77b4",
        "3e": "#ff7f0e",
        "4e": "#2ca02c",
        "2e_1n": "#d62728",
        "high_spin": "#9467bd",
        "default": "#7f7f7f"  # fallback color
    }

    x_positions = []
    mean_times = []
    std_times = []
    colors = []
    labels = []
    mesh_sizes = []

    current_x = 0
    system_boundaries = []

    for key, mesh_data in results.items():
        system_boundaries.append(current_x - 0.25)
        system_label = results[key][list(mesh_data.keys())[0]]["system_label"]

        for mesh_label, timing in mesh_data.items():
            x_positions.append(current_x)
            mean_times.append(timing["mean_ms"])
            std_times.append(timing["std_ms"])
            colors.append(color_map.get(key, color_map["default"]))
            labels.append(f"{system_label}\n{timing['mesh_resolution']}")
            mesh_sizes.append(timing["mesh_size"])
            current_x += 1

        system_boundaries.append(current_x - 0.75)  # Right boundary of system group
        current_x += 0.5  # Gap between systems

    bars = plt.bar(
        x_positions,
        mean_times,
        yerr=std_times,
        color=colors,
        alpha=0.85,
        capsize=5,
        ecolor='black',
        edgecolor='black',
        linewidth=0.8
    )

    for i, (x, y, size) in enumerate(zip(x_positions, mean_times, mesh_sizes)):
        plt.text(
            x,
            y + std_times[i] + max(mean_times) * 0.03,
            f"{size}",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='darkblue'
        )

    for boundary in system_boundaries[1:-1:2]:
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)

    plt.xticks(x_positions, labels, fontsize=9, rotation=30, ha='right')

    from matplotlib.patches import Patch
    legend_elements = []
    for key in results.keys():
        hilbert_dim = results[key][list(results[key].keys())[0]]["hilbert_dim"]
        system_label = results[key][list(results[key].keys())[0]]["system_label"]
        legend_elements.append(
            Patch(facecolor=color_map.get(key, color_map["default"]),
                  label=f"{system_label} (dim={hilbert_dim})")
        )

    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.xlabel("System Configuration & Mesh Resolution\n(mesh size in orientations shown above bars)",
               fontsize=12, fontweight='bold')
    plt.ylabel("Execution Time (ms)", fontsize=12, fontweight='bold')
    plt.title(
        f"EPR Spectrum Calculation Time Benchmark\n"
        f"Device: {device.type.upper()}, Dtype: {dtype}, "
        f"{len(x_positions)} configurations tested",
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim(0, max(mean_times) * 1.25)


def benchmark_several_configurations(
        config_creators: tp.Dict[str, tp.Tuple[str, tp.Callable]],
        mesh_configs: tp.Optional[tp.List[tp.Union[None, tp.Tuple[int, int]]]] = None,
        freq: float = 9.8e9,
        field_range: tp.Tuple[float, float] = (0.30, 0.40),
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        plot_results: bool = True,
        n_warmup: int = 10,
        n_iterations: int = 20,
) -> dict:
    """
    Parameters
    ----------
    config_creators : dict
        Dictionary mapping system keys to (label, creator_function) tuples.
        Example: {"2e": ("Two electrons", create_2_electrons_sample), ...}
    mesh_configs : list of (None or tuple), optional
        List of mesh configurations to test. Default tests coarse, medium, fine.
    freq : float, optional
        Microwave frequency in Hz. Default is 9.8 GHz (X-band).
    field_range : tuple of (float, float), optional
        Magnetic field range (min, max) in Tesla.
    device : torch.device, optional
        Computation device. Default is cpu.
    dtype : torch.dtype, optional
        Floating point precision.
    plot_results : bool, optional
        If True, generates a bar plot of timing results. Default is True.

    n_warmup : int, optional
        Number of warmup iterations (discarded from timing). Default is 10.
    n_iterations : int, optional
        Number of timed iterations. Default is 20.

    Returns
    -------
    results : dict
        Nested dictionary with timing results organized by system type and mesh.
    """
    if mesh_configs is None:
        mesh_configs = [(10, 10), (20, 20), (25, 25)]

    results = {}

    print(f"Benchmarking on {device.type.upper()} with {dtype}")
    print("=" * 70)

    for key, (label, creator_func) in config_creators.items():
        results[key] = {}
        print(f"\n{label} (Hilbert dim: ", end="")

        test_sample = creator_func(mesh=mesh_configs[0], device=device, dtype=dtype)
        hilbert_dim = test_sample.spin_system_dim
        print(f"{hilbert_dim})")

        for mesh in mesh_configs:
            mesh_label = f"{mesh[0]}x{mesh[1]}" if isinstance(mesh, tuple) else "default"
            print(f"  Mesh {mesh_label:12s} ... ", end="", flush=True)

            sample = creator_func(mesh=mesh, device=device, dtype=dtype)

            mean_ms, std_ms, _ = time_spectrum_calculation(
                sample,
                n_warmup=n_warmup,
                n_iterations=n_iterations,
                field_range=field_range,
                freq=freq,
            )

            if hasattr(sample.mesh, 'initial_size'):
                mesh_size = sample.mesh.initial_size[0]
            else:
                mesh_size = 1 if mesh is None else mesh[0] * mesh[1]

            results[key][mesh_label] = {
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "hilbert_dim": hilbert_dim,
                "mesh_size": mesh_size,
                "system_label": label,
                "mesh_resolution": mesh_label
            }

            print(f"{mean_ms:7.2f} ± {std_ms:5.2f} ms ({mesh_size} orientations)")

    if plot_results:
        _plot_benchmark_results(results, device, dtype)

    return results


def compare_benchmarks(benchmarks: tp.List[tp.Dict], plot_comparison=True) -> tp.Dict:
    """
    Compare multiple benchmark result dictionaries.

    Parameters
    ----------
    benchmarks : list of dict
        List of results dictionaries (each from benchmark_all_configurations or load_benchmark_results).
        Each dict has structure:
        {
            "system_key": {
                "mesh_label": {
                    "mean_ms": ...,
                    "std_ms": ...,
                    "hilbert_dim": ...,
                    "mesh_size": ...,
                    "system_label": ...,
                    "mesh_resolution": ...
                }
            }
        }

    Returns
    -------
    comparison : dict
        Dictionary containing:
        - 'summary': DataFrame with mean/std times per configuration across runs
        - 'speedups': Dict of speedup ratios (run0 vs runN) for each configuration
        - 'raw': Full DataFrame with all timing data
    """
    # Flatten all benchmarks into a single DataFrame
    records = []
    for run_idx, results in enumerate(benchmarks):
        for sys_key, mesh_data in results.items():
            for mesh_label, metrics in mesh_data.items():
                records.append({
                    "run": run_idx,
                    "system_key": sys_key,
                    "system_label": metrics["system_label"],
                    "mesh_resolution": mesh_label,
                    "mesh_size": metrics["mesh_size"],
                    "hilbert_dim": metrics["hilbert_dim"],
                    "mean_time_ms": metrics["mean_ms"],
                    "std_time_ms": metrics["std_ms"]
                })

    df = pd.DataFrame(records)

    summary = df.groupby(["system_key", "mesh_resolution", "hilbert_dim", "mesh_size"]).agg({
        "mean_time_ms": ["mean", "std", "min", "max"],
        "std_time_ms": "mean"
    })
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.round(2)

    speedups = {}
    if len(benchmarks) > 1:
        baseline = df[df["run"] == 0].set_index(["system_key", "mesh_resolution"])["mean_time_ms"]
        for run_idx in range(1, len(benchmarks)):
            run_times = df[df["run"] == run_idx].set_index(["system_key", "mesh_resolution"])["mean_time_ms"]
            ratios = (baseline / run_times).dropna()
            speedups[f"run0_vs_run{run_idx}"] = ratios.to_dict()

    comparison = {
        "summary": summary,
        "speedups": speedups,
        "raw": df
    }

    if plot_comparison:
        plot_benchmark_comparison(comparison)

    return comparison


def plot_benchmark_comparison(comparison: tp.Dict) -> None:
    """
    Plot comparison results from compare_benchmarks().

    Parameters
    ----------
    comparison : dict
        Output dictionary from compare_benchmarks() containing 'raw' DataFrame.

    """
    df = comparison["raw"]
    runs = sorted(df["run"].unique())
    configs = sorted(df.set_index(["system_key", "mesh_resolution"]).index.unique().tolist())

    # Create composite labels
    config_labels = [f"{row['system_label']}\n{row['mesh_resolution']}"
                     for _, row in df.drop_duplicates(["system_key", "mesh_resolution"]).iterrows()]

    x = np.arange(len(configs))
    width = 0.8 / len(runs)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Color scheme for runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    # Plot bars for each run
    for i, run_idx in enumerate(runs):
        run_data = df[df["run"] == run_idx].sort_values(["system_key", "mesh_resolution"])
        offset = (i - len(runs) / 2) * width + width / 2

        bars = ax.bar(
            x + offset,
            run_data["mean_time_ms"],
            width=width,
            label=f"Run {run_idx}",
            color=colors[i],
            alpha=0.85,
            edgecolor='black',
            linewidth=0.8
        )

        ax.errorbar(
            x + offset,
            run_data["mean_time_ms"],
            yerr=run_data["std_time_ms"],
            fmt='none',
            ecolor='black',
            capsize=3,
            alpha=0.6
        )

    for idx, (sys_key, mesh_res) in enumerate(configs):
        mesh_size = df[(df["system_key"] == sys_key) & (df["mesh_resolution"] == mesh_res)]["mesh_size"].iloc[0]
        ax.text(
            idx,
            ax.get_ylim()[1] * 0.98,
            f"{int(mesh_size)}",
            ha='center',
            va='top',
            fontsize=9,
            fontweight='bold',
            color='darkblue'
        )

    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=9, rotation=30, ha='right')
    ax.set_ylabel("Execution Time (ms)", fontsize=12, fontweight='bold')
    ax.set_xlabel("System Configuration & Mesh Resolution\n(mesh size in orientations shown above bars)",
                  fontsize=12, fontweight='bold')
    ax.set_title(f"Benchmark Comparison Across {len(runs)} Runs", fontsize=14, fontweight='bold', pad=15)
    ax.legend(title="Runs", fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    if len(runs) > 1 and "speedups" in comparison:
        for idx, (sys_key, mesh_res) in enumerate(configs):
            key = (sys_key, mesh_res)
            if key in comparison["speedups"].get("run0_vs_run1", {}):
                speedup = comparison["speedups"]["run0_vs_run1"][key]

                pos_x = idx + ((0 - len(runs) / 2) * width + width / 2 + (1 - len(runs) / 2) * width + width / 2) / 2
                ax.text(
                    pos_x,
                    ax.get_ylim()[1] * 0.02,
                    f"{speedup:.1f}x",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold',
                    color='green',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
                )

    plt.tight_layout()
    plt.show()