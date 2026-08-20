import copy
import os
import json
import warnings

from tqdm.auto import tqdm

import pathlib
from dataclasses import dataclass
import typing as tp
import math
from abc import ABC, abstractmethod

import numpy as np
import torch

import nevergrad as ng
import optuna

from ..spectra_processing import normalize_spectrum, normalize_spectrum2d
from . import objectives
from . import penalty_computations
from . import recreate_samplers
from optuna_dashboard import run_server


class TrialResult(tp.TypedDict):
    trial_number: int
    params: dict[str, float]
    delta: dict[str, float]
    loss: float
    distance: float


def print_trial_results(results: tp.Union[TrialResult, list[TrialResult]], max_params=None, precision=6) -> None:
    """Print trial results.

    :param results: Single trial dict or list of trial dicts
    :param max_params: Maximum number of parameters to display (None for
        all)
    :param precision: Number of decimal places for numeric values
    :return: None
    """
    if isinstance(results, dict):
        results = [results]

    for i, trial in enumerate(results):
        if i > 0:
            print("\n" + "=" * 80 + "\n")

        # Print header
        print(f"TRIAL #{trial['trial_number']}")
        print(f"index: {i}")
        print("-" * 40)
        print(f"Loss:     {trial['loss']:.{precision}f}")
        print(f"Distance: {trial['distance']:.{precision}f}")
        print()

        params = trial["params"]
        deltas = trial["delta"]
        param_names = list(params.keys())

        if max_params and len(param_names) > max_params:
            param_names = param_names[:max_params]
            truncated = True
        else:
            truncated = False

        print("PARAMETERS:")
        print("-" * 40)
        max_name_len = max(len(name) for name in param_names) if param_names else 0
        for param_name in param_names:
            value = params[param_name]
            delta = deltas.get(param_name, None)

            if isinstance(value, float):
                if abs(value) > 1000000:
                    value_str = f"{value:.{precision - 2}e}"
                else:
                    value_str = f"{value:.{precision}f}"
            else:
                value_str = str(value)

            if isinstance(delta, float):
                if abs(delta) > 1000000:
                    delta_str = f"{delta:+.{precision - 2}e}"
                else:
                    delta_str = f"{delta:+.{precision}f}"
            else:
                delta_str = str(delta)

            print(f"  {param_name:<{max_name_len}} = {value_str:>15} (Δ {delta_str})")
        if truncated:
            remaining = len(params) - max_params
            print(f"  ... and {remaining} more parameters")


def print_params(params: dict[str, float], max_params=None, precision=6) -> None:
    """
    :param params: the dict of parameter names and their values.

    :param max_params: maximum number of parameters
    :param precision: Number of decimal places for numeric values
    :return: None
    """

    param_names = list(params.keys())
    if max_params and len(param_names) > max_params:
        param_names = param_names[:max_params]
        truncated = True
    else:
        truncated = False

    print("PARAMETERS:")
    print("-" * 40)
    max_name_len = max(len(name) for name in param_names) if param_names else 0

    for param_name, in param_names:
        value = params[param_name]

        if isinstance(value, float):
            if abs(value) > 1000000:
                value_str = f"{value:.{precision - 2}e}"
            else:
                value_str = f"{value:.{precision}f}"
        else:
            value_str = str(value)

        print(f"  {param_name:<{max_name_len}} = {value_str:>15} ")

    if truncated:
        remaining = len(params) - max_params
        print(f"  ... and {remaining} more parameters")


@dataclass
class FitResult:
    best_params: tp.Dict[str, float]
    best_loss: float
    best_spectrum: tp.Optional[torch.Tensor]
    optimizer_info: tp.Dict

    def to_json_dict(self) -> dict:
        """Return a JSON‑compatible dict (spectrum is omitted, optimizer_info is sanitized)."""
        info = self.optimizer_info
        backend = info.get("backend")
        serialized_info = {"backend": backend}

        if backend == "optuna":
            study = info.get("study")
            if study is not None:
                serialized_info["trials"] = [
                    {
                        "trial_id": t.number,
                        "params": t.params,
                        "value": t.value,
                        "state": t.state.name if hasattr(t.state, "name") else str(t.state),
                    }
                    for t in study.trials
                ]
                serialized_info["best_trial_id"] = study.best_trial.number

        elif backend == "nevergrad":
            trials = info.get("trials", [])
            serialized_info["optimizer"] = info.get("optimizer")
            serialized_info["trials"] = [
                {
                    "trial_id": t._trial_id,
                    "params": t.params,
                    "value": t.value,
                }
                for t in trials
            ]

        else:
            for k, v in info.items():
                if k != "backend":
                    try:
                        json.dumps(v)
                        serialized_info[k] = v
                    except TypeError:
                        serialized_info[k] = repr(v)

        return {
            "best_params": self.best_params,
            "best_loss": self.best_loss,
            "optimizer_info": serialized_info,
        }

    def save(self, path: tp.Union[str, pathlib.Path], save_spectrum: bool = True) -> None:
        """Save FitResult to disk.

        :param path: Base filename (without extension). Creates ``path.json``
                     and optionally ``path_spectrum.pt``.
        :param save_spectrum: If True, save the best_spectrum tensor(s) as well.
        """
        json_path = path + ".json"
        with open(json_path, "w") as f:
            json.dump(self.to_json_dict(), f, indent=2)
        if save_spectrum and self.best_spectrum is not None:
            torch.save(self.best_spectrum, path + "_spectrum.pt")

    @classmethod
    def load(cls, path: str, load_spectrum: bool = True) -> "FitResult":
        """Load FitResult from disk."""
        json_path = path + ".json"
        with open(json_path, "r") as f:
            data = json.load(f)

        spectrum = None
        if load_spectrum:
            spec_path = path + "_spectrum.pt"
            try:
                spectrum = torch.load(spec_path)
            except FileNotFoundError:
                pass

        return cls(
            best_params=data["best_params"],
            best_loss=data["best_loss"],
            best_spectrum=spectrum,
            optimizer_info=data["optimizer_info"],
        )


@dataclass
class ExperementalParameters:
    best_params: tp.Dict[str, float]
    best_loss: float
    best_spectrum: tp.Optional[torch.Tensor]
    optimizer_info: tp.Dict


@dataclass
class NevergradTrial:
    params: tp.Dict[str, float]
    value: float
    _trial_id: int

    def __repr__(self):
        return f"_trial_id: {self._trial_id}, loss: {self.value}"

    def __str__(self):
        return f"_trial_id: {self._trial_id}, loss: {self.value}"


class TrialsTracker:
    def __init__(self):
        self.trials = []
        self._penalized_losses = []
        self._raw_losses = []
        self.step = 0

    def __call__(self, optimizer: ng.optimization.Optimizer,
                 candidate: ng.p.Instrumentation, loss: float):
        """Callback function called after each evaluation."""
        self.trials.append(candidate.value[0])
        self._penalized_losses.append(loss)
        self.step += 1

        if self.step % 10 == 0:
            print(f"Step {self.step}: Loss = {loss:.6f}")

    def override_last_raw_loss(self, raw_loss: float) -> None:
        """Call after __call__ to record the true loss for the most recent trial."""
        if len(self._raw_losses) < len(self.trials):
            self._raw_losses.append(raw_loss)
        elif self._raw_losses:
            self._raw_losses[-1] = raw_loss

    def _get_losses_for_output(self) -> list[float]:
        """Return raw losses if fully populated, else fall back to penalized."""
        if self._raw_losses and len(self._raw_losses) == len(self.trials):
            return self._raw_losses
        return self._penalized_losses

    def get_best_trial(self):
        """Get the trial with the lowest loss."""
        losses = self._get_losses_for_output()
        best_idx = np.argmin(losses)
        return {
            "_trial_id": best_idx + 1,
            "params": self.trials[best_idx],
            "value": losses[best_idx]
        }

    def get_all_trials(self):
        """Get all trials as a list of dictionaries."""
        losses = self._get_losses_for_output()
        return [
            {
                "_trial_id": i + 1,
                "params": trial,
                "value": loss
            }
            for i, (trial, loss) in enumerate(zip(self.trials, losses))
        ]


class Transform(tp.Protocol):
    def __call__(self, x: float) -> float:
        pass

    def inverse(self, y: float) -> float:
        pass


class LogTransform(Transform):
    def __call__(self, x: float) -> float:
        return math.log10(x)

    def inverse(self, y: float) -> float:
        return math.pow(10, y)


@dataclass
class ParamSpec:
    """Specification for a single scalar parameter.

    Attributes:
        name: parameter name

        bounds: (low, high) bounds for optimizer search (floats)

        default: optional default value to use for initialization

        transform: optional callable applied to a physical values to map them with optimizer/raw values
                   (for example, log-scales)

        vary: bool: Whether the parameter should vary or not.
        In the latter case, this is equivalent to specifying the parameter in fixed_parameters.
        If you don't plan to vary the parameter, then the more correct way is to specify it in fixed_parameters.
    """
    name: str
    bounds: tp.Tuple[float, float]
    default: tp.Optional[float] = None
    transform: tp.Optional[Transform] = None
    vary: bool = True

    def clip(self, x: float) -> float:
        """
        :param x: the value that should be clipped with respect to bounds
        :return: clipped values
        """
        lo, hi = self.bounds
        return float(min(max(x, lo), hi))

    def apply(self, x: float) -> float:
        """
        Gives the real value if transformed is skipped or return value transformed from raw/optimizer to physical space
        :param x: the vales given in optimizer space that should be transformed into physical space
        :return:
        """
        x = self.transform.inverse(x) if self.transform is not None else x
        return self.clip(x)

    def set_bounds(self, bounds: tp.Tuple[float, float]):
        """Update the bounds for this parameter spec."""
        self.bounds = bounds

    def raw_bounds(self) -> tp.Tuple[float, float]:
        """Bounds in the raw/search space that optimizers should sample from.
        Equal to `bounds` unless a transform is set, in which case they are
        the inverse-transformed physical bounds."""
        lo, hi = self.bounds
        if self.transform is not None:
            lo, hi = self.transform(lo), self.transform(hi)
        return lo, hi

    def raw_default(self) -> float:
        """Starting point in raw/search space corresponding to `default`
        (or the midpoint of `bounds` if no default was given)."""
        if self.default is not None:
            val = float(self.default)
        else:
            lo, hi = self.bounds
            val = 0.5 * (lo + hi)
        if self.transform is not None:
            val = self.transform(val)
        return val


class ParameterSpace:
    """
    A collection of fixed and varying scalar parameters.

    Each varying parameter is described by a :class:`ParamSpec`. Bounds and
    default values are always specified in the final (physical) space,
    regardless of whether a parameter has a transform attached — a
    transform only changes the space an optimizer searches internally.

    Example
    -------
    Define a space with two varying parameters: ``g_factor`` with plain
    bounds, and ``gauss`` with bounds searched  in log-space internally::

        specs = [
            ParamSpec(name="g_factor", bounds=(2.00, 2.01), default=2.005),
            ParamSpec(
                name="gauss",
                bounds=(1e-4, 1e-3),
                default=5e-4,
                transform=LogTransform(),
            ),
        ]
        space = ParameterSpace(specs)

        space["g_factor"]   # 2.005
        space["gauss"]      # 0.0005

    Both parameters are read and written in physical units; the log-space
    handling for ``gauss`` happens internally.
    """
    print_precision: int = 4

    def __init__(self, specs: tp.Sequence[ParamSpec],
                 fixed_params: tp.Optional[tp.Dict[str, float]] = None):
        """
        :param specs: The sequence of ParamSpec instances.

        The list include parameters that should be varied (if spec.vary = True. For more details
        see ParamSpec documentation)
        :param fixed_params: The parameters that are fixed during fit.
        """
        self.specs = list(specs)

        self.fixed_params: tp.Dict[str, float] = {} if fixed_params is None else dict(fixed_params)
        self.fixed_params.update({s.name: s.default for s in self.specs if not getattr(s, "vary")})

        self._varying_specs = [s for s in self.specs if getattr(s, "vary", True)]
        self.varying_names = [s.name for s in self._varying_specs]
        self.varying_params = {s.name: s.default for s in self._varying_specs}

        for name in list(self.fixed_params.keys()):
            if name in self.varying_names:
                idx = next(i for i, s in enumerate(self._varying_specs) if s.name == name)
                del self._varying_specs[idx]
                self.varying_names.remove(name)

    def __deepcopy__(self, memo):
        new_obj = type(self).__new__(type(self))

        new_obj.specs = copy.deepcopy(self.specs, memo)
        new_obj.fixed_params = copy.deepcopy(self.fixed_params, memo)
        new_obj._varying_specs = copy.deepcopy(self._varying_specs, memo)
        new_obj.varying_names = copy.deepcopy(self.varying_names, memo)
        new_obj.varying_params = copy.deepcopy(self.varying_params, memo)
        new_obj.print_precision = self.print_precision
        return new_obj

    def __getitem__(self, key: str):
        try:
            return self.fixed_params[key]
        except KeyError:
            try:
                return self.varying_params[key]
            except KeyError:
                raise KeyError(f"Key '{key}' not found in fixed_params or _varying_specs")

    def __setitem__(self, key: str, value: float):
        if key in self.fixed_params:
            self.fixed_params[key] = value
        elif key in self.varying_names:
            for spec in self._varying_specs:
                if spec.name == key:
                    spec.default = value
                    self.varying_params[key] = value
        else:
            raise KeyError(f"Key '{key}' not found in fixed_params or varying_params")

    def __dict__(self) -> dict[str, float]:
        return {**self.varying_params, **self.fixed_params}

    def __iter__(self):
        return iter(self.__dict__().items())

    def __repr__(self) -> str:
        """Print parameters space."""
        text = ""
        text += f"____Fixed parameters_____ \n"
        text += "-" * 40 + "\n"
        param_names = list(self.fixed_params.keys())
        for key, value in self.fixed_params.items():
            max_name_len = max(len(name) for name in param_names) if param_names else 0
            if isinstance(value, float):
                if abs(value) > 1000:
                    value_str = f"{value:.{self.print_precision - 2}e}"
                else:
                    value_str = f"{value:.{self.print_precision}f}"
            else:
                value_str = str(value)
            text += f"  {key:<{max_name_len}} = {value_str:>15}\n"

        text += "\n\n"
        text += f"______Varying parameters_____ \n"
        text += "-" * 40 + "\n"

        param_names = list(self.varying_names)
        for spec in self._varying_specs:
            max_name_len = max(len(name) for name in param_names) if param_names else 0

            value = spec.default
            name = spec.name
            (low, up) = spec.bounds

            if isinstance(value, float):
                if abs(value) > 1000:
                    value_str = f"{value:.{self.print_precision - 2}e}"
                else:
                    value_str = f"{value:.{self.print_precision}f}"
            else:
                value_str = str(value)

            if isinstance(low, float):
                if abs(low) > 1000000:
                    low = f"{low:+.{self.print_precision - 2}e}"
                else:
                    low = f"{low:+.{self.print_precision}f}"
            else:
                low = str(low)

            if isinstance(up, float):
                if abs(up) > 1000:
                    up = f"{up:+.{self.print_precision - 2}e}"
                else:
                    up = f"{up:+.{self.print_precision}f}"
            else:
                up = str(up)
            text += f"  {name:<{max_name_len}} = {value_str:>15}   (low:   {low}  up:   {up})\n"

        return text

    @classmethod
    def from_json_dict(cls, data: dict) -> "ParameterSpace":
        """Reconstruct a ParameterSpace from a dictionary."""
        specs = []
        transform_registry = {"LogTransform": LogTransform()}
        for s_dict in data["specs"]:
            transform = None
            if "transform" in s_dict:
                transform = transform_registry.get(s_dict["transform"])
            spec = ParamSpec(
                name=s_dict["name"],
                bounds=tuple(s_dict["bounds"]),
                default=s_dict.get("default"),
                transform=transform,
                vary=s_dict.get("vary", True),
            )
            specs.append(spec)
        return cls(specs, fixed_params=data.get("fixed_params"))

    def to_json_dict(self) -> dict:
        """Convert ParameterSpace to a JSON-serializable dictionary."""
        specs_data = []
        for s in self.specs:
            spec_dict = {
                "name": s.name,
                "bounds": list(s.bounds),
                "default": s.default,
                "vary": getattr(s, "vary", True),
            }
            if s.transform is not None:
                spec_dict["transform"] = s.transform.__class__.__name__
            specs_data.append(spec_dict)

        return {
            "specs": specs_data,
            "fixed_params": self.fixed_params,
        }

    def save(self, path: tp.Union[str, pathlib.Path]) -> None:
        """Save the parameter space to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_json_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ParameterSpace":
        """Load a parameter space from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_json_dict(data)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self, name: str, value: tp.Optional[float] = None):
        """Freeze a parameter by name.

        If value is provided, use it; otherwise use its default (or
        current) value.
        """
        if name not in self.varying_names:
            raise KeyError(name)
        spec = next(s for s in self.specs if s.name == name)

        if value is None:
            if spec.default is not None:
                value = float(spec.default)
            else:
                lo, hi = spec.bounds
                value = 0.5 * (lo + hi)
        self.fixed_params[name] = float(value)

        self._varying_specs = [s for s in self._varying_specs if s.name != name]
        self.varying_names = [s.name for s in self._varying_specs]
        self.varying_params = {s.name: s.default for s in self._varying_specs}

    def unfreeze(self, name: str):
        """Unfreeze a parameter previously frozen with `freeze` or
        fixed_params."""
        if name in self.fixed_params:
            del self.fixed_params[name]

        for s in self.specs:
            if s.name == name and s not in self._varying_specs and getattr(s, 'vary', True):
                self._varying_specs.append(s)
                self.varying_names.append(s.name)
        self.varying_params = {s.name: s.default for s in self._varying_specs}

    def vector_to_dict(self, vec: tp.Sequence[float]) -> tp.Dict[str, float]:
        """Convert an optimizer vector of raw/optimizer space values
        (ordered only over *varying* params).

        into a full parameter dict that includes fixed parameters.
        """
        if len(vec) != len(self._varying_specs):
            raise ValueError(f"Expected vector of length {len(self._varying_specs)}, got {len(vec)}")
        out = dict(self.fixed_params)  # start with fixed
        for s, v in zip(self._varying_specs, vec):
            out[s.name] = s.apply(float(v))
        return out

    def varying_vector_to_dict(self, vec: tp.Sequence[float]) -> tp.Dict[str, float]:
        """Convert an optimizer vector raw/optimizer space values
        (ordered only over *varying* params).

        into a full parameter dict that includes fixed parameters.
        """
        if len(vec) != len(self._varying_specs):
            raise ValueError(f"Expected vector of length {len(self._varying_specs)}, got {len(vec)}")
        out = {}
        for s, v in zip(self._varying_specs, vec):
            out[s.name] = s.apply(float(v))
        return out

    def dict_to_vector(self, params: tp.Dict[str, float]) -> np.ndarray:
        return np.array([params[n] for n in self.varying_names], dtype=float)

    def _set_single_bounds(self, param_name: str, bounds: tp.Tuple[float, float]):
        """
        :param param_name: name of the varying parameter.

        :param bounds: new bounds of the parameter
        :return: None
        """
        if param_name not in self.varying_names:
            raise KeyError(f"Parameter {param_name} not found in varying parameter space")

        low, high = bounds
        if low >= high:
            raise ValueError(f"Invalid bounds: low ({low}) must be less than high ({high})")

        for spec in self.specs:
            if spec.name == param_name:
                spec.set_bounds(bounds)
                break

    def set_default(self, params: dict[str, float]):
        """
        :param params: the dict of parameters.

        Set default value for parameters given in params
        :return:
        """
        for key, value in params.items():
            if key in self.fixed_params:
                self.fixed_params[key] = value
            elif key in self.varying_params:
                for spec in self._varying_specs:
                    if spec.name == key:
                        spec.default = value
                        self.varying_params[spec.name] = value
                        break
            else:
                raise KeyError(f"Key '{key}' not found in fixed_params or _varying_specs")

    def reduce_bounds(self, names: tp.Optional[str] = None, alpha: float = 0.2):
        """Reduces bounds of varying parameters.

        If bounds was (a, b) and default value c than the new bounds are:
        delta = (b-a)*alpha
        new_bounds = (c - delta, c+delta)
        :param names: names of parameters to reduce
        :param alpha: reducing coefficient
        :return: None
        """
        if names is None:
            names = self.varying_names
        for name in names:
            for spec in self._varying_specs:
                if spec.name == name:
                    default = spec.default
                    low, up = spec.bounds
                    delta = (up - low) * alpha

                    new_low = max(low, default-delta / 2)
                    new_up = min(up, default+delta / 2)
                    spec.bounds = (new_low, new_up)
                    break

    def set_bounds(self, bounds_dict: tp.Dict[str, tp.Tuple[float, float]]):
        """
        :param bounds_dict: the dict with names of parameters and their new bounds.

        :return: None
        """
        for param_name, bounds in bounds_dict.items():
            self._set_single_bounds(param_name, bounds)

    def suggest_optuna(self, trial) -> tp.Dict[str, float]:
        out = dict(self.fixed_params)
        for s in self._varying_specs:
            lo, hi = s.raw_bounds()
            val = trial.suggest_float(s.name, lo, hi)
            out[s.name] = s.apply(val)
        return out

    def get_optuna_distributions(self) -> tp.Dict[str, optuna.distributions.FloatDistribution]:
        return {s.name: optuna.distributions.FloatDistribution(*s.raw_bounds())
                for s in self._varying_specs}

    def instrument_nevergrad(self) -> ng.p.Instrumentation:
        params = []
        for s in self._varying_specs:
            lo, hi = s.raw_bounds()
            params.append(ng.p.Scalar(lower=lo, upper=hi, init=s.raw_default()))
        return ng.p.Instrumentation(*params)


def convert_backend_kwargs(
    backend: str,
    kwargs: tp.Dict[str, tp.Any]
) -> tp.Dict[str, tp.Any]:
    """Converts common keyword arguments between optimization backends and filters out incompatible ones.

    :param backend: Target backend name ('optuna' or 'nevergrad').
    :param kwargs: Dictionary of keyword arguments to process.
    :return: A cleaned dictionary containing only parameters valid for the target backend.
        Equivalent parameters are automatically renamed. Parameters specific to the
        alternate backend or unrecognized by both are removed and a warning is issued.
    """
    target = backend.lower().replace("ng", "nevergrad")
    source = "nevergrad" if target == "optuna" else "optuna"

    _OPTUNA_KWARGS = {
        "sampler", "pruner", "n_trials", "timeout", "n_jobs",
        "callbacks", "show_progress_bar", "max_concurrent_trials", "run_dashboard"
    }
    _NEVERGRAD_KWARGS = {
        "optimizer", "budget", "num_workers", "timeout",
        "callback", "executor", "with_progress", "batch_mode"
    }

    _KNOWN = {"optuna": _OPTUNA_KWARGS, "nevergrad": _NEVERGRAD_KWARGS}
    _CONVERSION_MAP = {
        "optuna": {
            "n_trials": "budget",
            "sampler": "optimizer",
            "n_jobs": "num_workers",
            "show_progress_bar": "with_progress",
            "callbacks": "callback"
        },
        "nevergrad": {
            "budget": "n_trials",
            "optimizer": "sampler",
            "num_workers": "n_jobs",
            "with_progress": "show_progress_bar",
            "callback": "callbacks"
        }
    }

    target_known = _KNOWN[target]
    source_known = _KNOWN[source]
    mapping = _CONVERSION_MAP[source]

    converted = {}
    unrecognized = []

    for key, value in kwargs.items():
        if key in target_known:
            converted[key] = value
        elif key in source_known and key in mapping:
            converted[mapping[key]] = value
        else:
            converted[key] = value
            unrecognized.append(key)

    if unrecognized:
        warnings.warn(
            f"Unrecognized {len(unrecognized)} keyword argument(s) specific to: {unrecognized}",
            UserWarning,
            stacklevel=2
        )
    return converted


class CWSpectraSimulator:
    """Example of CW spectra simulator."""
    def __init__(self,
                 sample_updator: tp.Callable[[dict[str, float], tp.Any], tp.Any],
                 spectra_creator: tp.Callable[[tp.Any, torch.Tensor], torch.Tensor], *args):
        """
        :param sample_updator: Callable object that updates sample.

        :param spectra_creator: Callable object that creates spectra
        :param args:
        """
        self.sample_updator = sample_updator
        self.spectra_creator = spectra_creator
        self.args = args

    def __call__(self, fields: torch.Tensor, params: dict[str, float]):
        """
        :param fields: magnetic fields in Tesla units.

        :param params: parameters of param space
        :return:
        """
        sample = self.sample_updator(params, *self.args)
        return self.spectra_creator(sample, fields)


class BaseSpectrumFitter(ABC):
    """Base class for spectrum fitting."""
    __available_optimizer__ = {
        "nevergrad": sorted(ng.optimizers.registry.keys()),
        "optuna": [
            optuna.integration.BoTorchSampler,
            optuna.samplers.RandomSampler,
            optuna.samplers.TPESampler,
            optuna.samplers.BruteForceSampler,
            optuna.samplers.GridSampler,
            optuna.samplers.CmaEsSampler,
            optuna.samplers.NSGAIISampler,
            optuna.samplers.NSGAIIISampler,
        ],
    }

    def __init__(
        self,
        param_space: ParameterSpace,
        spectra_simulator: tp.Callable,
        norm_mode: str = "integral",
        objective=objectives.MSEObjective(),
        weights: tp.Optional[list[float]] = None,
        penalty: penalty_computations.RepulsivePenalty = penalty_computations.RepulsivePenalty(),
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        :param param_space: The object of ParameterSpace class where all varying parameters are included.
        :param spectra_simulator: Callable that takes x-data and parameters and returns simulated spectra
        :param norm_mode: Norm mode to fit data. 'integral' / 'max'
        :param objective: Used objective function. It should be an inheritor of objectives.BaseObjective
        :param weights: The weights for multi-spectra fit. Default is None
        :param penalty: The Class to manage penalty for the repulsion from the local minima
        :param device: Device for computation
        """
        self.norm_mode = norm_mode
        self._simulate_callable = spectra_simulator
        self.param_space = param_space
        self._objective = objective

        self.penalty = penalty
        self.x_exp = None
        self.y_exp = None
        self.multisample = False
        self.weights = None
        self._loss_normalization = None
        self._n_data_points = None
        self._proportional_to_mse = False

    @abstractmethod
    def _set_experimental(self, *args, **kwargs):
        """Process experimental data and set  them self.x_exp, self.y_exp,
        self.multisample."""
        pass

    @abstractmethod
    def _simulate_single_spectrum(self, params: dict[str, float], **kwargs) -> torch.Tensor:
        """Simulate spectrum from set of parameters.

        :param params: Full parameter dictionary
        :return: Normalized single simulated spectrum
        """
        pass

    @abstractmethod
    def _simulate_spectral_set(self, params: dict[str, float], **kwargs) -> list[torch.Tensor]:
        """Simulate set of spectra from set of parameters.

        :param params: Full parameter dictionary
        :return: List of normalized simulated spectra
        """
        pass

    @property
    def device(self):
        return self.y_exp.device if hasattr(self.y_exp, "device") else self.y_exp[0].device

    @property
    def dtype(self):
        return self.y_exp.dtype if hasattr(self.y_exp, "dtype") else self.y_exp[0].dtype

    @property
    def proportional_to_mse(self):
        return self._proportional_to_mse

    def _get_loss_norm(self):
        if self.multisample:
            return [self._objective(torch.zeros_like(y), y).reciprocal() for y in self.y_exp]
        else:
            return self._objective(torch.zeros_like(self.y_exp), self.y_exp).reciprocal()

    def simulate_spectroscopic_data(self, params: tp.Dict[str, float], **kwargs) ->\
            tp.Union[list[torch.Tensor], torch.Tensor]:
        """
        :param params: fict of parameter names: parameter values.

        The names of parameters are names from param_space.
        Example:
        fitter.simulate_spectroscopic_data(dict(param_space))
        :param kwargs:
        :return: Simulated spectra - list or single spectra
        """
        if self.multisample:
            model = self._simulate_spectral_set(params, **kwargs)
        else:
            model = self._simulate_single_spectrum(params, **kwargs)
        return model

    def simulate_spectra_from_trial_params(self, trial_params: tp.Dict[str, float], **kwargs) ->\
            tp.Union[list[torch.Tensor], torch.Tensor]:
        """
        :param trial_params: Simulate spectra from parameters given as trial_params (only varied parameters).

        As fixed_parameters the parameters from self.param_space are used
        :param kwargs:
        :return: Simulated spectra - list or single spectra
        """
        return self.simulate_spectroscopic_data({**self.param_space.fixed_params, **trial_params}, **kwargs)

    def _loss_from_params(self, params: tp.Dict[str, tp.Union[float, torch.Tensor]], **kwargs) -> torch.Tensor:
        """Compute model - experiment residuals as a torch.Tensor."""
        with torch.no_grad():
            if self.multisample:
                models = self._simulate_spectral_set(params, **kwargs)
                loss = sum(self.weights[idx] * self._loss_normalization[idx] * self._objective(
                    models[idx], self.y_exp[idx]) for idx in range(len(models)))
            else:
                model = self._simulate_single_spectrum(params, **kwargs)
                loss = self._loss_normalization * self._objective(model, self.y_exp)
            return loss

    def _loss_from_params_random(
        self,
        params: tp.Dict[str, float],
        rng: tp.Optional[torch.Generator] = None,
        **kwargs
    ) -> tp.Union[torch.Tensor, tp.List[torch.Tensor]]:

        device = self.device
        dtype = self.dtype

        if rng is None:
            rng = torch.Generator(device=device).manual_seed(42)

        with torch.no_grad():
            if self.multisample:
                models = self._simulate_spectral_set(params, **kwargs)
                loss = torch.tensor(0, dtype=dtype, device=device)
                for model_idx in range(len(models)):
                    sample_idx = torch.randint(
                        0, len(self.y_exp[model_idx]), (len(self.y_exp[model_idx]),),
                        generator=rng, device=device, dtype=dtype
                    )

                    loss += self.weights[model_idx] * self._loss_normalization[model_idx] * self._objective(
                    models[model_idx][sample_idx], self.y_exp[model_idx][sample_idx]
                    )
            else:
                sample_idx = torch.randint(
                    0, len(self.y_exp), (len(self.y_exp),), generator=rng,
                    device=device, dtype=dtype
                )

                model = self._simulate_single_spectrum(params, **kwargs)
                loss = self._loss_normalization * self._objective(model[sample_idx], self.y_exp[sample_idx])
            return loss

    def _tracker_to_trials(self, trials_tracker: TrialsTracker) -> list[NevergradTrial]:
        trials_all_results = trials_tracker.get_all_trials()
        ng_trials = [
            NevergradTrial(params=self.param_space.varying_vector_to_dict(trial["params"]),
                           _trial_id=trial["_trial_id"],
                           value=trial["value"]
                           ) for trial in trials_all_results
        ]
        return ng_trials

    def fit_optuna(
            self,
            show_progress: bool,
            seed: tp.Optional[int],
            return_best_spectrum: bool,

            n_trials: int = 300,
            timeout: tp.Optional[float] = None,
            n_jobs: int = 1,
            sampler: tp.Optional[optuna.samplers.BaseSampler] = None,
            study_name: tp.Optional[str] = None,
            run_dashboard: bool = True,
            **kwargs,
    ) -> FitResult:
        """Fit spectra using Optuna.

        Requires optuna to be installed.
        """

        def loss_function(trial):
            p = self.param_space.suggest_optuna(trial)
            loss = self._loss_from_params(p, **kwargs)
            return loss

        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if run_dashboard:
            storage = optuna.storages.InMemoryStorage()
            study = optuna.create_study(direction="minimize", sampler=sampler,
                                        study_name=study_name, load_if_exists=True, storage=storage)
            study.optimize(
                loss_function, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, show_progress_bar=show_progress)
            run_server(storage)
        else:
            study = optuna.create_study(direction="minimize", sampler=sampler,
                                        study_name=study_name, load_if_exists=True)
            study.optimize(
                loss_function, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, show_progress_bar=show_progress)

        best_params = {k: float(v) for k, v in study.best_params.items()}
        best_spec = None
        if return_best_spectrum:
            best_spec = self.simulate_spectroscopic_data({**self.param_space.fixed_params, **best_params}, **kwargs)
        return FitResult(best_params, float(study.best_value), best_spec, {"backend": "optuna", "study": study})

    def fit_nevergrad(
            self,
            show_progress: bool,
            seed: tp.Optional[int],
            return_best_spectrum: bool,

            budget: int = 200,
            optimizer: str = "TwoPointsDE",
            track_trials: bool = True,
            **kwargs,
    ) -> FitResult:
        """Fit spectra using Nevergrad (if installed)."""
        if ng is None:
            raise RuntimeError("Nevergrad is required for fit_nevergrad but not installed")

        instr = self.param_space.instrument_nevergrad()
        if seed is not None:
            ng.optimizers.registry.seed(seed)
        opt = ng.optimizers.registry[optimizer](parametrization=instr, budget=budget)

        def _loss_from_tuple(*args):
            params = self.param_space.vector_to_dict(args)
            return self._loss_from_params(params).item()

        if show_progress:
            progress_bar = ng.callbacks.ProgressBar()
            progress_bar.update_frequency = 25
            opt.register_callback("tell", progress_bar)

        trials_tracker = None
        if track_trials:
            trials_tracker = TrialsTracker()
            opt.register_callback("tell", trials_tracker)

        recommendation = opt.minimize(_loss_from_tuple)
        x = recommendation.value
        best_params = self.param_space.varying_vector_to_dict(x[0])
        best_spec = None
        if return_best_spectrum:
            best_spec = self.simulate_spectroscopic_data({**self.param_space.fixed_params, **best_params})

        trials = None
        if track_trials:
            trials = self._tracker_to_trials(trials_tracker)

        return FitResult(
            best_params, self._loss_from_params({**self.param_space.fixed_params, **best_params}), best_spec,
            {"backend": "nevergrad", "optimizer": optimizer, "trials": trials}
        )

    def _get_raw_loss_from_vector(self, vec: tp.Sequence[float]) -> float:
        """Compute raw (unpenalized) loss from optimizer vector."""
        params = self.param_space.vector_to_dict(vec)
        return self._loss_from_params(params).item()

    def _compute_penalty_for_vector(self, vec: tp.Sequence[float], penalty_force: float) -> float:
        """Compute penalty for a single parameter vector using self.penalty."""
        X = np.asarray(vec).reshape(1, -1)
        penalty_val = self.penalty.compute_penalty(X)
        if isinstance(penalty_val, np.ndarray):
            return float(penalty_val.item()) * penalty_force
        return float(penalty_val) * penalty_force

    def _update_penalty_state(self, X_history: np.ndarray,
                              losses: np.ndarray,
                              param_dicts: list[dict[str, float]]) -> None:
        """Update internal penalty state using historical raw losses."""
        self.penalty.update(X_history, losses, param_dicts)

    def _warmstart_study_optuna(
            self,
            study: optuna.study.Study,
            param_dicts: list[dict[str, float]],
            raw_losses: list[float],
            penalty_idx: np.ndarray,
            penalty_force: float = 1.0,
            use_penalty: bool = True
    ) -> None:
        """Replay historical trials on a fresh Optuna study with updated penalized losses."""
        if not param_dicts:
            return
        param_names = param_dicts[0].keys()
        X_history = np.array([
            [params[name] for name in param_names] for params in param_dicts
        ])[:, penalty_idx]
        distributions = self.param_space.get_optuna_distributions()
        if use_penalty:
            penalties = self.penalty.compute_penalty(X_history).flatten() * penalty_force
            for params, raw_loss, penalty in zip(param_dicts, raw_losses, penalties):
                loss = raw_loss + float(penalty)
                trial = optuna.trial.create_trial(
                    params=params,
                    value=loss,
                    distributions=distributions
                )
                trial.set_user_attr("raw_loss", raw_loss)
                study.add_trial(trial)
        else:
            for params, raw_loss in zip(param_dicts, raw_losses):
                loss = raw_loss
                trial = optuna.trial.create_trial(
                    params=params,
                    value=loss,
                    distributions=distributions
                )
                trial.set_user_attr("raw_loss", raw_loss)
                study.add_trial(trial)

    def _warmstart_optimizer_ng(
            self,
            optimizer: ng.optimizers.base.Optimizer,
            parametrization: ng.p.Parameter,
            param_vectors: list[tp.Sequence[float]],
            raw_losses: list[float],
            penalty_idx: np.ndarray,
            penalty_force: float = 1.0,
    ) -> None:
        """Replay historical trials on a fresh optimizer with updated penalized losses."""
        if not param_vectors:
            return
        X_history = np.array([np.asarray(v).flatten() for v in param_vectors])[:, penalty_idx]
        penalties = self.penalty.compute_penalty(X_history)

        penalties = penalties.flatten() * penalty_force

        for params_vec, raw_loss, penalty in zip(param_vectors, raw_losses, penalties):
            candidate = parametrization.spawn_child()
            candidate.value = (tuple(params_vec), {})
            optimizer.tell(candidate, raw_loss + float(penalty))

    def fit_nevergrad_penalty(
            self,
            show_progress: bool,
            seed: tp.Optional[int],
            return_best_spectrum: bool,
            budget: int = 200,
            optimizer: str = "TwoPointsDE",
            track_trials: bool = True,

            penalty_names: tp.Optional[list[str]] = None,
            update_penalty_every: int = 20,
            restart_every: int = 60,
            penalty_force: float = 1.0,
            **kwargs,
    ) -> FitResult:
        """
        Fit spectra using Nevergrad with dynamic penalty updates.

        The penalty is updated periodically using historical raw losses,
        and the optimizer is warm-started with recomputed penalized losses.
        Final results report RAW losses (not penalized).

        :param show_progress: If True, show a progress bar via Nevergrad callback.
        :param seed: Random seed for the Nevergrad optimizer.
        :param return_best_spectrum: If True, simulate and return the spectrum
            at the best parameters.
        :param budget: Total number of function evaluations (optimization budget).
        :param optimizer: Name of the Nevergrad optimizer to use
            (e.g., "TwoPointsDE", "NGOpt").
        :param track_trials: If True, record all trials for later analysis.
        :param penalty_names: The name os parameters which should add penalty for the local minima.
            That is this defines the dimensions for penalty loss. Default is all variable names

        :param update_penalty_every: Recompute the penalty state every N evaluations.
        :param restart_every: Create a fresh optimizer and warm-start it every N evaluations.
        :param penalty_force: Multiplier applied to the penalty term.
        :param kwargs: Extra arguments passed to `simulate_spectroscopic_data`.
        :return: FitResult containing best parameters, raw loss, optional spectrum,
            and additional info.
        """
        if ng is None:
            raise RuntimeError("Nevergrad is required for fit_nevergrad_penalty but not installed")
        if penalty_names is None:
            penalty_names = self.param_space.varying_names
            penalty_idx = np.arange(0, len(penalty_names))
        else:
            varying_names = self.param_space.varying_names
            penalty_idx = np.array([idx for idx in range(len(varying_names)) if varying_names[idx] in penalty_names])

        instr = self.param_space.instrument_nevergrad()
        if seed is not None:
            ng.optimizers.registry.seed(seed)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        raw_losses: list[float] = []
        param_vectors: list[tp.Sequence[float]] = []

        trials_tracker = TrialsTracker() if track_trials else None
        progress_bar = ng.callbacks.ProgressBar() if show_progress else None
        if progress_bar:
            progress_bar.update_frequency = 25

        opt = ng.optimizers.registry[optimizer](parametrization=instr, budget=budget)
        if trials_tracker:
            opt.register_callback("tell", trials_tracker)
        if progress_bar:
            opt.register_callback("tell", progress_bar)

        for step in range(budget):
            candidate = opt.ask()
            param_vec = np.array(candidate.value[0])

            raw_loss = self._get_raw_loss_from_vector(param_vec)

            penalty_val = self._compute_penalty_for_vector(param_vec[penalty_idx], penalty_force)
            penalized_loss = raw_loss + penalty_val

            opt.tell(candidate, penalized_loss)

            raw_losses.append(raw_loss)
            param_vectors.append(param_vec)

            if trials_tracker:
                trials_tracker.override_last_raw_loss(raw_loss)

            if (step + 1) % update_penalty_every == 0 and step + 1 < budget:
                X_hist = np.array([np.asarray(v).flatten() for v in param_vectors])
                losses_hist = np.array(raw_losses, dtype=float)
                param_dicts = [self.param_space.vector_to_dict(v) for v in param_vectors]
                self._update_penalty_state(X_hist[:, penalty_idx], losses_hist, param_dicts)

            if (step + 1) % restart_every == 0 and step + 1 < budget:
                opt = ng.optimizers.registry[optimizer](parametrization=instr, budget=budget)
                self._warmstart_optimizer_ng(opt, instr, param_vectors, raw_losses, penalty_idx, penalty_force)
                if trials_tracker:
                    opt.register_callback("tell", trials_tracker)
                if progress_bar:
                    opt.register_callback("tell", progress_bar)

        if raw_losses:
            best_idx = int(np.argmin(raw_losses))
            best_params = self.param_space.varying_vector_to_dict(param_vectors[best_idx])
            best_raw_loss = raw_losses[best_idx]
        else:
            best_params = dict(self.param_space.varying_params)
            best_raw_loss = float('inf')

        best_spec = None
        if return_best_spectrum:
            best_spec = self.simulate_spectroscopic_data(
                {**self.param_space.fixed_params, **best_params}, **kwargs)

        trials = None
        if track_trials and trials_tracker:
            trials = self._tracker_to_trials(trials_tracker)

        self.penalty.clear()
        return FitResult(
            best_params, best_raw_loss, best_spec,
            {
                "backend": "nevergrad",
                "optimizer": optimizer,
                "trials": trials,
                "update_penalty_every": update_penalty_every,
                "restart_every": restart_every,
                "eval_count": len(raw_losses)
            }
        )

    def fit_optuna_penalty(
            self,
            show_progress: bool,
            seed: tp.Optional[int],
            return_best_spectrum: bool,
            n_trials: int = 300,

            n_jobs: int = 1,
            sampler: tp.Optional[optuna.samplers.BaseSampler] = None,
            study_name: tp.Optional[str] = None,
            run_dashboard: bool = True,

            penalty_names: tp.Optional[list[str]] = None,
            update_penalty_every: int = 20,
            restart_every: int = 60,
            penalty_force: float = 1.0,
            **kwargs,
    ) -> FitResult:
        """
        Fit spectra using Optuna with dynamic penalty updates.

        The penalty is updated periodically using historical raw losses,
        and the study is warm‑started with recomputed penalized losses.
        Final results report RAW losses (not penalized).

        :param show_progress: If True, show a tqdm progress bar.
        :param seed: Random seed for the sampler (if the sampler supports it).
        :param return_best_spectrum: If True, simulate and return the spectrum
            at the best parameters.
        :param n_trials: Total number of Optuna trials (evaluation budget).
        :param n_jobs: Number of parallel jobs. Currently only n_jobs=1 is supported
            (no parallelism). A warning is raised if n_jobs > 1.
        :param sampler: Optuna sampler (e.g., TPESampler). Defaults to
            TPESampler(multivariate=True, seed=seed).
        :param study_name: Name of the Optuna study (passed to `create_study`).
        :param run_dashboard: If True, launch the Optuna dashboard after optimization.
            Requires `optuna-dashboard` installed.

        :param penalty_names: The name os parameters which should add penalty for the local minima.
            That is this defines the dimensions for penalty loss. Default is all variable names
        :param update_penalty_every: Recompute the penalty state every N trials.
        :param restart_every: Create a fresh study and warm-start it every N trials.
        :param penalty_force: Multiplier applied to the penalty term.
        :param kwargs: Extra arguments passed to `simulate_spectroscopic_data`.
        :return: FitResult containing best parameters, raw loss, optional spectrum,
            and additional info (including the final study and raw losses).
        """
        if optuna is None:
            raise RuntimeError("Optuna is required for fit_optuna_penalty but not installed")

        if n_jobs != 1:
            raise NotImplementedError("For the loss with penalty, Optuna supports only one job")

        def _suggest_optuna(study: optuna.Study, **kwargs):
            trial = study.ask()
            params = self.param_space.suggest_optuna(trial)
            raw_loss = self._loss_from_params(params, **kwargs).item()
            items = list(params.items())
            start = len(self.param_space.fixed_params)

            params = dict(
                items[start:]
            )
            return trial, params, raw_loss

        def _create_study(sampler: optuna.samplers.BaseSampler):
            if show_progress:
                storage = optuna.storages.InMemoryStorage()
                study = optuna.create_study(
                    direction="minimize",
                    sampler=sampler,
                    study_name=study_name,
                    load_if_exists=True,
                    storage=storage,
                )
            else:
                study = optuna.create_study(
                    direction="minimize",
                    sampler=sampler,
                    study_name=study_name,
                    load_if_exists=True,
                )
            return study

        def _restart_study(study: optuna.Study):
            sampler = study.sampler
            sampler = recreate_samplers.recreate_sampler_without_history(sampler)
            study_name = study.study_name
            if show_progress:
                storage = study._storage
                optuna.delete_study(study_name=study_name, storage=storage)
                study = optuna.create_study(
                    direction="minimize",
                    sampler=sampler,
                    study_name=study_name,
                    storage=storage,
                )
            else:
                study = optuna.create_study(
                    direction="minimize",
                    sampler=sampler,
                    study_name=study_name,
                )

            study.sampler = sampler
            return study

        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        else:
            if seed is not None and hasattr(sampler, "seed"):
                sampler.seed = seed

        if penalty_names is None:
            penalty_names = self.param_space.varying_names
            penalty_idx = np.arange(0, len(penalty_names))
        else:
            varying_names = self.param_space.varying_names
            penalty_idx = np.array([idx for idx in range(len(varying_names)) if varying_names[idx] in penalty_names])

        raw_losses: list[float] = []
        param_dicts: list[dict[str, float]] = []
        param_vectors: list[np.ndarray] = []

        pbar = None

        study = _create_study(sampler)
        if show_progress:
            pbar = tqdm(total=n_trials, desc="Optuna penalty optimization")

        for step in range(n_trials):
            trial, params, raw_loss = _suggest_optuna(study, **kwargs)
            vec = self.param_space.dict_to_vector(params)
            penalty_val = self._compute_penalty_for_vector(vec[penalty_idx], penalty_force)
            penalized_loss = raw_loss + penalty_val

            study.tell(trial, penalized_loss)

            raw_losses.append(raw_loss)
            param_dicts.append(params)
            param_vectors.append(vec)

            if pbar:
                pbar.update(1)
                line = f"best trial: {np.argmin(raw_losses)}; best_rwa_loss: {min(raw_losses)}"
                pbar.set_postfix_str(line)

            if (step + 1) % update_penalty_every == 0 and step + 1 < n_trials:
                X_hist = np.array([v.flatten() for v in param_vectors])
                losses_hist = np.array(raw_losses, dtype=float)
                self._update_penalty_state(X_hist[:, penalty_idx], losses_hist, param_dicts)

            if (step + 1) % restart_every == 0 and step + 1 < n_trials:
                new_study = _restart_study(study)

                self._warmstart_study_optuna(new_study, param_dicts, raw_losses, penalty_idx, penalty_force)
                study = new_study

        if pbar:
            pbar.close()

        if raw_losses:
            best_idx = int(np.argmin(raw_losses))
            best_params = param_dicts[best_idx]
            best_raw_loss = raw_losses[best_idx]
        else:
            best_params = dict(self.param_space.varying_params)
            best_raw_loss = float("inf")

        best_spec = None
        if return_best_spectrum:
            best_spec = self.simulate_spectroscopic_data(
                {**self.param_space.fixed_params, **best_params}, **kwargs
            )

        new_study = _restart_study(study)
        self._warmstart_study_optuna(new_study, param_dicts, raw_losses, penalty_idx, penalty_force, use_penalty=False)
        study = new_study

        if run_dashboard:
            run_server(study._storage)

        self.penalty.clear()
        return FitResult(
            best_params,
            best_raw_loss,
            best_spec,
            {
                "backend": "optuna",
                "study": study,
                "raw_losses": raw_losses,
                "param_dicts": param_dicts,
                "update_penalty_every": update_penalty_every,
                "restart_every": restart_every,
                "eval_count": len(raw_losses),
            },
        )

    def fit(
            self,
            backend: str = "optuna",
            seed: tp.Optional[int] = None,
            show_progress: bool = True,
            return_best_spectrum: bool = True,

            use_penalty: bool = False,

            penalty_names: tp.Optional[list[str]] = None,
            update_penalty_every: int = 20,
            restart_every: int = 60,
            penalty_force: float = 1.0,

            **backend_kwargs,
    ) -> FitResult:
        """All fitting methods can be viewed in
        ''SpectrumFitter.__available_optimizer__.''

        :param backend: optuna / nevergrad. Sets which library should be used to fit data.
            Optuna supports not as many methods as nevergrad but they are quite powerful. Default fitting method is TPE.
            TPE has quite high exploration abilities and not as dramatic speed of work as Bayesian models.
            After the initial fitting process it is recommended to reduce
            the bounds and continue fitting with any method of convex optimization from Nevergrad:
            For example, with COBYLA.

        :param seed: Random seed for the optimizer/sampler.

        :param show_progress: If True, display a progress bar.

        :param return_best_spectrum: If True, simulate and return the spectrum at the best
            parameters.

        :param use_penalty: If True, use the penalty‑based variant of the optimizer.
            This enables dynamic penalty updates and warm‑restarts.

        :param penalty_names: The name os parameters which should add penalty for the local minima.
            That is this defines the dimensions for penalty loss. Default is all variable names

        :param penalty_force: Multiplier applied to the penalty term (only when
            ``use_penalty=True``).

        :param update_penalty_every: Recompute penalty state every N evaluations (only when
            ``use_penalty=True``).

        :param restart_every: Create a fresh optimizer/study every N evaluations and
            warm‑start it with historical data (only when ``use_penalty=True``).

        :param backend_kwargs: The kwargs of fit settings described in optuna / nevergrad library
            NOTE! Optuna and Nevergrad have different backend parameters.
            We have saved the initial naming from these libraries

            Key differences:
                                        optuna                                   nevergrad
            ----------------------------------------------------------------------------------------------------
            method type          optuna.samplers.BaseSampler                    str object
            ----------------------------------------------------------------------------------------------------
            number of iterations      n_trials                                    budget
            ----------------------------------------------------------------------------------------------------

        :return: None
        """
        method = backend.lower()
        backend_kwargs = convert_backend_kwargs(backend, backend_kwargs)
        if method == "optuna":
            if use_penalty:
                return self.fit_optuna_penalty(
                    seed=seed,
                    show_progress=show_progress,
                    return_best_spectrum=return_best_spectrum,
                    penalty_names=penalty_names,
                    penalty_force=penalty_force,
                    update_penalty_every=update_penalty_every,
                    restart_every=restart_every,
                    **backend_kwargs,
                )
            else:
                return self.fit_optuna(
                    seed=seed,
                    show_progress=show_progress,
                    return_best_spectrum=return_best_spectrum,
                    **backend_kwargs,
                )
        if method in ("nevergrad", "ng"):
            if use_penalty:
                return self.fit_nevergrad_penalty(
                    seed=seed,
                    show_progress=show_progress,
                    penalty_names=penalty_names,
                    return_best_spectrum=return_best_spectrum,
                    penalty_force=penalty_force,
                    update_penalty_every=update_penalty_every,
                    restart_every=restart_every,
                    **backend_kwargs,
                )
            else:
                return self.fit_nevergrad(
                    seed=seed,
                    show_progress=show_progress,
                    return_best_spectrum=return_best_spectrum,
                    **backend_kwargs,
                )
        raise ValueError(f"Unknown fit method: {method}")

    def save_state(self, path: tp.Union[str, pathlib.Path]) -> None:
        """Save the parameter space and experimental data.

        Subclasses should call `super().save_state(path)` and then save their
        specific experimental tensors.
        """
        state = {
            "param_space": self.param_space.to_json_dict(),
            "norm_mode": self.norm_mode,
            "objective": self._objective.__class__.__name__,
            "multisample": self.multisample,
            "weights": self.weights.tolist() if self.weights is not None else None,
        }
        with open(path + "_state.json", "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def _load_state_common(cls, path: tp.Union[str, pathlib.Path], device: torch.device, dtype: torch.dtype):
        """Load common state from JSON file and return components."""
        with open(path + "_state.json", "r") as f:
            state = json.load(f)

        param_space = ParameterSpace.from_json_dict(state["param_space"])
        objective_name = state["objective"]
        objective_cls = objectives.OBJECTIVE_REGISTRY.get(objective_name)
        if objective_cls is None:
            raise ValueError(f"Unknown objective class: {objective_name}")
        objective = objective_cls()

        weights = state.get("weights")
        if weights is not None:
            weights = torch.tensor(weights, dtype=dtype, device=device)

        return param_space, objective, state["norm_mode"], state["multisample"], weights


class SpectrumFitter(BaseSpectrumFitter):
    """General fitter for spectra.

    The user must provide either a `simulate_spectrum_callable` that maps a
    parameter dict -> torch.Tensor (spectrum on the same B-grid), or override
    the `simulate_spectrum` method in a subclass.

    Typical usage:
      - construct with B grid, experimental spectrum (np or torch), device
      - provide parameter specs
      - call fit(method='optuna'|'nevergrad')
    """
    __available_optimizer__ = {"nevergrad": sorted(ng.optimizers.registry.keys()),
                               "optuna": [optuna.integration.BoTorchSampler,
                                          optuna.samplers.RandomSampler,
                                          optuna.samplers.TPESampler,
                                          optuna.samplers.BruteForceSampler,
                                          optuna.samplers.GridSampler,
                                          optuna.samplers.CmaEsSampler,
                                          optuna.samplers.NSGAIISampler,
                                          optuna.samplers.NSGAIIISampler,
                                          ]
                               }

    def __init__(
            self,
            x_exp: tp.Union[np.ndarray, torch.Tensor, tp.List[tp.Union[np.ndarray, torch.Tensor]]],
            y_exp: tp.Union[np.ndarray, torch.Tensor, tp.List[tp.Union[np.ndarray, torch.Tensor]]],
            param_space: ParameterSpace,
            spectra_simulator: tp.Callable[
                [tp.Union[tp.List[torch.Tensor], torch.Tensor], tp.Dict[str, float], tp.Dict],
                tp.Union[torch.Tensor, tp.List[torch.Tensor]]
            ],
            norm_mode: str = "integral",
            objective=objectives.MSEObjective(),
            weights: tp.Optional[tp.List[float]] = None,
            penalty: penalty_computations.RepulsivePenalty = penalty_computations.RepulsivePenalty(),
            device: torch.device = torch.device("cpu"),
            dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        :param x_exp: Experimental x-axis data.

        It can be magnetic field (T), time (s)
            It is possible to pass a list for multi-object fit
        :param y_exp: Experimental y-axis data.
        :param param_space: The object of ParameterSpace class where all varying parameters are included
        :param spectra_simulator: Any callable object that takes x_data and parameters and returns simulated
            spectra or list of simulated spectra.
            It is highly recommended for all new parameters to use update methods:
            sample.update(new_params) or spec_creator.update_config(config)

            Example:
            class CWSpectraSimulator:
                def __init__(self,
                             sample_updator: tp.Callable[[dict[str, float], tp.Any], tp.Any],
                             spectra_creator: tp.Callable[[tp.Any, torch.Tensor], torch.Tensor], *args):
                    self.sample_updator = sample_updator
                    self.spectra_creator = spectra_creator
                    self.args = args

                def __call__(self, fields: torch.Tensor, params: dict[str, float]):
                    sample = self.sample_updator(params, *self.args)
                    return self.spectra_creator(sample, fields)

        :param norm_mode: Norm mode to fit data. 'integral' / 'max'
        :param device: Device for computation
        :param objective: Used objective function. It should be an inheritor of objectives.BaseObjective
        :param weights: The weights for multi-data fit. Default is None
        :param penalty: The Class to manage penalty for the repulsion from the local minima
        """
        super().__init__(param_space, spectra_simulator, norm_mode, objective, weights, penalty, device, dtype)

        self.x_exp, self.y_exp, self.multisample = self._set_experimental(x_exp, y_exp, device, dtype)

        if self.multisample and (weights is None):
            self.weights = torch.ones(len(self.x_exp), dtype=dtype, device=device) / len(self.x_exp)
        elif weights is not None:
            self.weights = torch.tensor(weights, dtype=dtype, device=device)
            self.weights = self.weights / self.weights.sum()
        else:
            self.weights = None
        self._loss_normalization = self._get_loss_norm()

        if self.multisample:
            self._n_data_points = sum(len(x) for x in self.x_exp)
            self._proportional_to_mse = False
        else:
            self._n_data_points = len(self.x_exp)
            self._proportional_to_mse = self._objective.LOSS_PROPORTIONAL_TO_MSE

    def _set_experimental(
            self, x_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
                  y_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
                  device: torch.device, dtype: torch.dtype) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, bool]:
        """Set expereimental given parameter.

        :param x_exp: Experimental x-axis data. It can be magnetic field
            (T), time (s)
        :param y_exp: Experimental y-axis data.
        :return: x_exp, y_exp in appropriate format and flag that is
            multisample fitting data
        """
        if isinstance(x_exp, list):
            if len(x_exp) != len(y_exp):
                raise ValueError("The number of x array and experimental arrays must be the same")
            else:
                x_exp = [torch.tensor(b, dtype=dtype, device=device) for b in x_exp]
                y_exp = [torch.tensor(y, dtype=dtype, device=device) for y in y_exp]
                for idx, b in enumerate(x_exp):
                    y_exp[idx] = normalize_spectrum(b, y_exp[idx], mode=self.norm_mode)
                multisample = True
        else:
            x_exp = torch.tensor(x_exp, dtype=dtype, device=device)
            y_exp = torch.tensor(y_exp, dtype=dtype, device=device)
            y_exp = normalize_spectrum(x_exp, y_exp, mode=self.norm_mode)
            multisample = False
        return x_exp, y_exp, multisample

    def _simulate_single_spectrum(self, params: tp.Dict[str, float], **kwargs) -> torch.Tensor:
        return normalize_spectrum(self.x_exp, self._simulate_callable(self.x_exp, params, **kwargs), mode=self.norm_mode)

    def _simulate_spectral_set(self, params: tp.Dict[str, float], **kwargs) -> list[torch.Tensor]:
        models = self._simulate_callable(self.x_exp, params, **kwargs)
        for idx in range(len(models)):
            models[idx] = normalize_spectrum(self.x_exp[idx], models[idx], mode=self.norm_mode)
        return models

    def save_state(self, path: str) -> None:
        """Save the fitter state, including experimental 1D data."""
        super().save_state(path)
        torch.save(
            {
                "x_exp": self.x_exp,
                "y_exp": self.y_exp,
            },
            path + "_exp.pt",
        )

    @classmethod
    def load_state(
        cls,
        path: tp.Union[str, pathlib.Path],
        spectra_simulator: tp.Callable,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> "SpectrumFitter":
        """
        Reconstruct a SpectrumFitter from saved state.

        :param path: Base path (without extension) used in `save_state`.
        :param spectra_simulator: The same (or equivalent) simulator callable.
        :param device: torch device (if None, use CPU).
        :param dtype: torch dtype.
        :return: A new SpectrumFitter instance.
        """
        param_space, objective, norm_mode, multisample, weights = cls._load_state_common(
            path, device, dtype
        )

        exp_data = torch.load(path + "_exp.pt", map_location=device)
        x_exp = exp_data["x_exp"]
        y_exp = exp_data["y_exp"]

        if isinstance(x_exp, list):
            x_exp = [t.to(device=device, dtype=dtype) for t in x_exp]
            y_exp = [t.to(device=device, dtype=dtype) for t in y_exp]
        else:
            x_exp = x_exp.to(device=device, dtype=dtype)
            y_exp = y_exp.to(device=device, dtype=dtype)

        return cls(
            x_exp=x_exp,
            y_exp=y_exp,
            param_space=param_space,
            spectra_simulator=spectra_simulator,
            norm_mode=norm_mode,
            objective=objective,
            weights=weights.tolist() if weights is not None else None,
            device=device,
            dtype=dtype,
        )


class Spectrum2DFitter(BaseSpectrumFitter):
    """Spectrum Fitter for 2D data.

    y_exp should be 2d array, x1_exp and x2_exp are axis
    """
    def __init__(
            self,
            x1_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
            x2_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
            y_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
            param_space: ParameterSpace,
            spectra_simulator: tp.Callable[
                [tp.Union[list[torch.Tensor], torch.Tensor],
                 tp.Union[list[torch.Tensor], torch.Tensor],
                 tp.Dict[str, float], tp.Dict],
                tp.Union[torch.Tensor, list[torch.Tensor]]
            ],
            norm_mode: str = "integral",
            objective=objectives.MSEObjective(),
            weights: list[float] = None,
            penalty: penalty_computations.RepulsivePenalty = penalty_computations.RepulsivePenalty(),
            device: torch.device = torch.device("cpu"),
            dtype: torch.dtype = torch.float32,
    ):
        """
        :param x1_exp: Experimental x1-axis data.

        It can be magnetic field (T), time (s),
            It is possible to pass a list for multi-object fit
        :param x2_exp: Experimental x2-axis data. It can be magnetic field (T), time (s),
            It is possible to pass a list for multi-object fit

        :param y_exp: Experimental y-axis data.
        :param param_space: The object of ParameterSpace class where all varying parameters are included
        :param spectra_simulator: Any callable object that takes x_data and parameters and returns simulated
            spectra or list of simulated spectra.
            It is highly recommended for all new parameters to use update methods:
            sample.update(new_params) or spec_creator.update_config(config)

            Example:
            class CWSpectraSimulator:
                def __init__(self,
                             sample_updator: tp.Callable[[dict[str, float], tp.Any], tp.Any],
                             spectra_creator: tp.Callable[[tp.Any, torch.Tensor], torch.Tensor], *args):
                    self.sample_updator = sample_updator
                    self.spectra_creator = spectra_creator
                    self.args = args

                def __call__(self, fields: torch.Tensor, params: dict[str, float]):
                    sample = self.sample_updator(params, *self.args)
                    return self.spectra_creator(sample, fields)

        :param norm_mode: Norm mode to fit data. 'integral' / 'max'
        :param device: Device for computation
        :param objective: Used objective function. It should be an inheritor of objectives.BaseObjective
        :param weights: The weights for multi-data fit. Default is None
        :param penalty: The Class to manage penalty for the repulsion from the local minima
        """
        super().__init__(param_space, spectra_simulator, norm_mode, objective, weights, penalty, device, dtype)
        self.x1_exp, self.x2_exp, self.y_exp, self.multisample = self._set_experimental(
            x1_exp, x2_exp, y_exp, device, dtype)

        if self.multisample and (weights is None):
            self.weights = torch.ones(len(self.x1_exp), dtype=dtype, device=device) / len(self.x1_exp)
        elif weights is not None:
            self.weights = torch.tensor(weights, dtype=dtype, device=device)
            self.weights = self.weights / self.weights.sum()
        else:
            self.weights = None

        self._loss_normalization = self._get_loss_norm()

        if self.multisample:
            self._n_data_points = sum(len(x) for x in self.x1_exp)
            self._proportional_to_mse = False

        else:
            self._n_data_points = len(self.x1_exp)
            self._proportional_to_mse = self._objective.LOSS_PROPORTIONAL_TO_MSE


    def _set_experimental(
            self, x1_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
                  x2_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
                  y_exp: tp.Union[tp.Union[np.ndarray, torch.Tensor], list[tp.Union[np.ndarray, torch.Tensor]]],
                  device: torch.device, dtype: torch.dtype) ->\
            tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Set experimental given parameter.

        :param x1_exp: Experimental x1-axis data. It can be magnetic
            field (T), time (s)
        :param x2_exp: Experimental x1-axis data. It can be magnetic
            field (T), time (s)
        :param y_exp: Experimental y-axis data.
        :return: x1_exp, x2_exp, y_exp in appropriate format and flag
            that is multisample fitting data
        """
        if isinstance(x1_exp, list) and isinstance(x2_exp, list):
            if (len(x1_exp) != len(y_exp)) or (len(x2_exp) != len(y_exp)):
                raise ValueError("The number of x1 and x2 array and experimental arrays must be the same")
            else:
                x1_exp = [torch.tensor(b, dtype=dtype, device=device) for b in x1_exp]
                x2_exp = [torch.tensor(b, dtype=dtype, device=device) for b in x2_exp]
                y_exp = [torch.tensor(y, dtype=dtype, device=device) for y in y_exp]
                for idx, (b1, b2) in enumerate(zip(x1_exp, x2_exp)):
                    y_exp[idx] = normalize_spectrum2d(b1, b2, y_exp[idx], mode=self.norm_mode)
                multisample = True
        else:
            x1_exp = torch.tensor(x1_exp, dtype=dtype, device=device)
            x2_exp = torch.tensor(x2_exp, dtype=dtype, device=device)
            y_exp = torch.tensor(y_exp, dtype=dtype, device=device)
            y_exp = normalize_spectrum2d(x1_exp, x2_exp, y_exp, mode=self.norm_mode)
            multisample = False

        return x1_exp, x2_exp, y_exp, multisample

    def _simulate_single_spectrum(self, params: tp.Dict[str, float], **kwargs) -> torch.Tensor:
        return normalize_spectrum2d(self.x1_exp, self.x2_exp, self._simulate_callable(
            self.x1_exp, self.x2_exp, params, **kwargs), mode=self.norm_mode)

    def _simulate_spectral_set(self, params: tp.Dict[str, float], **kwargs) -> list[torch.Tensor]:
        models = self._simulate_callable(self.x1_exp, self.x2_exp, params, **kwargs)
        for idx in range(len(models)):
            models[idx] = normalize_spectrum2d(self.x1_exp[idx], self.x2_exp[idx], models[idx], mode=self.norm_mode)
        return models

    def save_state(self, path: tp.Union[str, pathlib.Path]) -> None:
        """Save the fitter state, including experimental 2D data."""
        super().save_state(path)
        torch.save(
            {
                "x1_exp": self.x1_exp,
                "x2_exp": self.x2_exp,
                "y_exp": self.y_exp,
            },
            path + "_exp.pt",
        )

    @classmethod
    def load_state(
        cls,
        path: tp.Union[str, pathlib.Path],
        spectra_simulator: tp.Callable,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> "Spectrum2DFitter":
        """
        Reconstruct a Spectrum2DFitter from saved state.

        :param path: Base path (without extension) used in `save_state`.
        :param spectra_simulator: The same (or equivalent) simulator callable.
        :param device: torch device (if None, use CPU).
        :param dtype: torch dtype.
        :return: A new Spectrum2DFitter instance.
        """
        param_space, objective, norm_mode, multisample, weights = cls._load_state_common(
            path, device, dtype
        )

        exp_data = torch.load(path + "_exp.pt", map_location=device)
        x1_exp = exp_data["x1_exp"]
        x2_exp = exp_data["x2_exp"]
        y_exp = exp_data["y_exp"]

        if isinstance(x1_exp, list):
            x1_exp = [t.to(device=device, dtype=dtype) for t in x1_exp]
            x2_exp = [t.to(device=device, dtype=dtype) for t in x2_exp]
            y_exp = [t.to(device=device, dtype=dtype) for t in y_exp]
        else:
            x1_exp = x1_exp.to(device=device, dtype=dtype)
            x2_exp = x2_exp.to(device=device, dtype=dtype)
            y_exp = y_exp.to(device=device, dtype=dtype)

        return cls(
            x1_exp=x1_exp,
            x2_exp=x2_exp,
            y_exp=y_exp,
            param_space=param_space,
            spectra_simulator=spectra_simulator,
            norm_mode=norm_mode,
            objective=objective,
            weights=weights.tolist() if weights is not None else None,
            device=device,
            dtype=dtype,
        )


class SpectrumCompositeFitter:
    """
    Composite fitter for simultaneously fitting multiple spectroscopic datasets
    with a shared parameter space.

    This class aggregates several :class:`BaseSpectrumFitter` instances, computes
    their individual losses, and combines them into a weighted sum. It provides a
    unified :meth:`fit` interface that delegates to the same backends (Optuna,
    Nevergrad) as the base fitter.

    All sub‑fitters must share the identical :class:`ParameterSpace` (the same
    varying parameter names, bounds, and fixed values). The combined loss is
    simply the weighted sum of the losses reported by each sub‑fitter.

    Attributes
    ----------
    param_space : ParameterSpace
        The common parameter space (extracted from the first fitter).
    fitters : List[BaseSpectrumFitter]
        The list of sub‑fitters.
    weights : List[float]
        The weights applied to each sub‑fitter's loss.
    """
    def __init__(
        self,
        fitters: tp.Sequence[BaseSpectrumFitter],
        weights: tp.Optional[tp.Sequence[float]] = None,
    ) -> None:
        """
        Initialise the composite fitter.

        :param fitters: Sequence of :class:`BaseSpectrumFitter` instances that
            will contribute to the combined loss.
        :param weights: Optional weights for each fitter. If ``None``, equal
            weights are used.
        :raises ValueError: If no fitters are provided, if their parameter
            spaces differ, or if the number of weights mismatches the number of
            fitters.
        """
        if not fitters:
            raise ValueError("At least one fitter must be provided.")

        first_ps = fitters[0].param_space
        for i, f in enumerate(fitters[1:], start=2):
            if f.param_space != first_ps:
                raise ValueError(
                    f"Fitter {i} has a different parameter space than the first fitter."
                )
        self.param_space = first_ps
        self.fitters = list(fitters)

        n = len(self.fitters)
        if weights is None:
            self.weights = [1.0 / n] * n
        else:
            if len(weights) != n:
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match number of fitters ({n})."
                )
            self.weights = list(weights)

        self._n_data_points = sum(fitter._n_data_points for fitter in fitters)
        self._proportional_to_mse = False

    @property
    def device(self):
        return self.fitters[0].device

    @property
    def dtype(self):
        return self.fitters[0].dtype

    @property
    def proportional_to_mse(self):
        return self._proportional_to_mse


    def _loss_from_params(self, params: tp.Dict[str, float], **kwargs) -> torch.Tensor:
        """
        Compute the combined loss from all sub‑fitters.

        The loss is a weighted sum of the individual losses returned by each
        fitter's :meth:`_loss_from_params` method.

        :param params: Full parameter dictionary (including fixed parameters).
        :param kwargs: Additional keyword arguments forwarded to each sub‑fitter.
        :return: Scalar loss tensor.
        """

        total_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for fitter, w in zip(self.fitters, self.weights):
            total_loss = total_loss + w * fitter._loss_from_params(params, **kwargs)
        return total_loss

    def _loss_from_params_random(
            self,
            params: tp.Dict[str, float],
            rng: tp.Optional[np.random.Generator] = None,
            **kwargs
    ) -> tp.Union[torch.Tensor, tp.List[torch.Tensor]]:
        if rng is None:
            rng = torch.Generator(device=self.device).manual_seed(42)

        total_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for fitter, w in zip(self.fitters, self.weights):
            total_loss = total_loss + w * fitter._loss_from_params_random(params, rng, **kwargs)
        return total_loss

    def _tracker_to_trials(self, trials_tracker: TrialsTracker) -> tp.List[NevergradTrial]:
        """
        Convert Nevergrad trial records to a list of :class:`NevergradTrial` objects.

        :param trials_tracker: The :class:`TrialsTracker` instance that collected
            trial data.
        :return: List of :class:`NevergradTrial` objects.
        """
        trials_all_results = trials_tracker.get_all_trials()
        ng_trials = [
            NevergradTrial(
                params=self.param_space.varying_vector_to_dict(trial["params"]),
                _trial_id=trial["_trial_id"],
                value=trial["value"],
            )
            for trial in trials_all_results
        ]
        return ng_trials

    def fit(
            self,
            backend: str = "optuna",
            seed: tp.Optional[int] = None,
            show_progress: bool = True,
            return_best_spectrum: bool = True,

            use_penalty: bool = False,

            penalty_names: tp.Optional[list[str]] = None,
            update_penalty_every: int = 20,
            restart_every: int = 60,
            penalty_force: float = 1.0,

            **backend_kwargs,
    ) -> FitResult:
        """All fitting methods can be viewed in
        ''SpectrumFitter.__available_optimizer__.''

        :param backend: optuna / nevergrad. Sets which library should be used to fit data.
            Optuna supports not as many methods as nevergrad but they are quite powerful. Default fitting method is TPE.
            TPE has quite high exploration abilities and not as dramatic speed of work as Bayesian models.
            After the initial fitting process it is recommended to reduce
            the bounds and continue fitting with any method of convex optimization from Nevergrad:
            For example, with COBYLA.

        :param seed: Random seed for the optimizer/sampler.

        :param show_progress: If True, display a progress bar.

        :param return_best_spectrum: If True, simulate and return the spectrum at the best
            parameters.

        :param use_penalty: If True, use the penalty‑based variant of the optimizer.
            This enables dynamic penalty updates and warm‑restarts.

        :param penalty_names: The name os parameters which should add penalty for the local minima.
            That is this defines the dimensions for penalty loss. Default is all variable names

        :param update_penalty_every: Recompute penalty state every N evaluations (only when
            ``use_penalty=True``).

        :param restart_every: Create a fresh optimizer/study every N evaluations and
            warm‑start it with historical data (only when ``use_penalty=True``).

        :param penalty_force: Multiplier applied to the penalty term (only when
            ``use_penalty=True``).

        :param backend_kwargs: The kwargs of fit settings described in optuna / nevergrad library
            NOTE! Optuna and Nevergrad have different backend parameters.
            We have saved the initial naming from these libraries

            Key differences:
                                        optuna                                   nevergrad
            ----------------------------------------------------------------------------------------------------
            method type          optuna.samplers.BaseSampler                    str object
            ----------------------------------------------------------------------------------------------------
            number of iterations      n_trials                                    budget
            ----------------------------------------------------------------------------------------------------

        :return: None
        """
        method = backend.lower()
        backend_kwargs = convert_backend_kwargs(backend, backend_kwargs)
        if method == "optuna":
            if use_penalty:
                raise NotImplementedError("Complex Fitter doesn't support penalty parameters")
            else:
                return self.fit_optuna(
                    seed=seed,
                    show_progress=show_progress,
                    return_best_spectrum=return_best_spectrum,
                    **backend_kwargs,
                )
        if method in ("nevergrad", "ng"):
            if use_penalty:
                raise NotImplementedError("Complex Fitter doesn't support penalty parameters")
            else:
                return self.fit_nevergrad(
                    seed=seed,
                    show_progress=show_progress,
                    return_best_spectrum=return_best_spectrum,
                    **backend_kwargs,
                )
        raise ValueError(f"Unknown fit method: {method}")

    def fit_optuna(
        self,
        show_progress: bool,
        seed: tp.Optional[int],
        return_best_spectrum: bool,
        n_trials: int = 300,
        timeout: tp.Optional[float] = None,
        n_jobs: int = 1,
        sampler: tp.Optional["optuna.samplers.BaseSampler"] = None,
        study_name: tp.Optional[str] = None,
        run_dashboard: bool = True,
        **kwargs,
    ) -> FitResult:
        """
        Internal method: fit using Optuna.

        :param show_progress: Show Optuna progress bar.
        :param seed: Random seed for sampler.
        :param return_best_spectrum: Whether to return best spectra.
        :param n_trials: Number of Optuna trials.
        :param timeout: Stop study after the given number of seconds.
        :param n_jobs: Number of parallel jobs.
        :param sampler: Optuna sampler instance.
        :param study_name: Name of the Optuna study.
        :param run_dashboard: If ``True``, launch an Optuna dashboard after
            optimisation.
        :param kwargs: Additional arguments passed to the loss function.
        :return: :class:`FitResult`
        """
        def loss_function(trial):
            p = self.param_space.suggest_optuna(trial)
            return self._loss_from_params(p, **kwargs).item()

        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if run_dashboard:
            storage = optuna.storages.InMemoryStorage()
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                study_name=study_name,
                load_if_exists=True,
                storage=storage,
            )
            study.optimize(
                loss_function,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                show_progress_bar=show_progress,
            )
            run_server(storage)
        else:
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                study_name=study_name,
                load_if_exists=True,
            )
            study.optimize(
                loss_function,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                show_progress_bar=show_progress,
            )

        best_params = {k: float(v) for k, v in study.best_params.items()}
        best_spec = None
        if return_best_spectrum:
            full_params = {**self.param_space.fixed_params, **best_params}
            best_spec = [f.simulate_spectroscopic_data(full_params, **kwargs) for f in self.fitters]

        return FitResult(
            best_params,
            float(study.best_value),
            best_spec,
            {"backend": "optuna", "study": study},
        )

    def fit_nevergrad(
        self,
        show_progress: bool,
        seed: tp.Optional[int],
        return_best_spectrum: bool,
        budget: int = 200,
        optimizer: str = "TwoPointsDE",
        track_trials: bool = True,
        **kwargs,
    ) -> FitResult:
        """
        Internal method: fit using Nevergrad.

        :param show_progress: Show Nevergrad progress bar.
        :param seed: Random seed.
        :param return_best_spectrum: Whether to return best spectra.
        :param budget: Number of function evaluations.
        :param optimizer: Name of the Nevergrad optimizer.
        :param track_trials: Whether to record trial history.
        :param kwargs: Additional arguments passed to the loss function.
        :return: :class:`FitResult`
        :raises RuntimeError: If Nevergrad is not installed.
        """
        instr = self.param_space.instrument_nevergrad()
        if seed is not None:
            ng.optimizers.registry.seed(seed)
        opt = ng.optimizers.registry[optimizer](parametrization=instr, budget=budget)

        def _loss_from_tuple(*args):
            params = self.param_space.vector_to_dict(args)
            return self._loss_from_params(params, **kwargs).item()

        if show_progress:
            progress_bar = ng.callbacks.ProgressBar()
            progress_bar.update_frequency = 25
            opt.register_callback("tell", progress_bar)

        trials_tracker = None
        if track_trials:
            trials_tracker = TrialsTracker()
            opt.register_callback("tell", trials_tracker)

        recommendation = opt.minimize(_loss_from_tuple)
        x = recommendation.value
        best_params = self.param_space.varying_vector_to_dict(x[0])

        best_spec = None
        if return_best_spectrum:
            full_params = {**self.param_space.fixed_params, **best_params}
            best_spec = [f.simulate_spectroscopic_data(full_params, **kwargs) for f in self.fitters]

        trials = None
        if track_trials:
            trials = self._tracker_to_trials(trials_tracker)

        return FitResult(
            best_params,
            self._loss_from_params({**self.param_space.fixed_params, **best_params}, **kwargs).item(),
            best_spec,
            {"backend": "nevergrad", "optimizer": optimizer, "trials": trials},
        )

    def save_state(self, path: str) -> None:
        """
        Save the composite fitter state.

        Each sub‑fitter is saved in a subdirectory named `fitter_<i>`.
        """
        os.makedirs(path, exist_ok=True)

        meta = {
            "num_fitters": len(self.fitters),
            "weights": self.weights,
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        for i, fitter in enumerate(self.fitters):
            sub_path = os.path.join(path, f"fitter_{i}")
            fitter.save_state(sub_path)

    @classmethod
    def load_state(
        cls,
        path: str,
        spectra_simulators: tp.List[tp.Callable],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> "SpectrumCompositeFitter":
        """
        Reconstruct a SpectrumCompositeFitter from saved state.

        :param path: Directory path used in `save_state`.
        :param spectra_simulators: List of simulator callables, one per sub‑fitter.
        :param device: torch device (if None, use CPU).
        :param dtype: torch dtype.
        :return: A new SpectrumCompositeFitter instance.
        """
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        fitters = []
        for i in range(meta["num_fitters"]):
            sub_path = os.path.join(path, f"fitter_{i}")
            exp_path = sub_path + "_exp.pt"
            exp_data = torch.load(exp_path, map_location="cpu")
            if "x1_exp" in exp_data:
                fitter_cls = Spectrum2DFitter
            else:
                fitter_cls = SpectrumFitter
            fitter = fitter_cls.load_state(
                sub_path,
                spectra_simulator=spectra_simulators[i],
                device=device,
                dtype=dtype,
            )
            fitters.append(fitter)

        return cls(fitters, weights=meta["weights"])
