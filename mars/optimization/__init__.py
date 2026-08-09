"""
mars.optimization – Parameter fitting and space exploration for EPR spectra

This package provides a comprehensive framework for fitting simulated EPR
spectra to experimental data using state‑of‑the‑art optimisation backends.
It supports:

  • **Parameter spaces** – define variable and fixed parameters with bounds
  • **Fitting backends** – Optuna (TPE, CMA‑ES, etc.) and Nevergrad
    (TwoPointsDE, NGOpt, COBYLA, …)
  • **1D and 2D spectra** – fit CW, time‑resolved, and multi‑dimensional data
  • **Composite fitting** – simultaneously fit multiple datasets with shared
    parameters
  • **Penalty‑based optimisation** – avoid local minima by dynamic repulsion
  • **Space exploration** – after fitting, discover diverse local minima and
    analyse the parameter landscape

Main components
---------------
ParameterSpace          – container for varying and fixed parameters
ParamSpec               – specification for a single parameter
BaseSpectrumFitter      – abstract base for all fitters
SpectrumFitter          – fit 1D spectra (e.g., CW EPR)
Spectrum2DFitter        – fit 2D spectra (e.g., field‑time maps)
SpectrumCompositeFitter – combine multiple fitters with a shared parameter space
FitResult               – stores the best parameters, loss, and optimisation history
SpaceSearcher           – extract diverse local minima from a fit result

Examples
--------
1. Define a parameter space for a simple CW fit:

>>> from mars.optimization import ParameterSpace, ParamSpec
>>> param_space = ParameterSpace([
...     ParamSpec("g_iso", bounds=(1.9, 2.1), default=2.0),
...     ParamSpec("linewidth", bounds=(0.1, 5.0), default=1.0, vary=True),
... ], fixed_params={"temperature": 300})

2. Create a spectrum simulator (see also `mars.spectra_manager`):

>>> import torch
>>> from mars import spectra_manager, spin_model
>>>
>>> # Build spin system and sample
>>> g_tensor = spin_model.Interaction(components=[2.0, 2.01, 2.02])
>>> system = spin_model.SpinSystem(electrons=[0.5], g_tensors=[g_tensor])
>>> sample = spin_model.MultiOrientedSample(base_spin_system=system, gauss=5e-4)
>>>
>>> # Simulator callable: updates sample and returns spectrum
>>> def simulator(fields, params):
...     sample = get_sample(params)
...     return spectra_manager.StationarySpectra(
...         freq=9.7e9, sample=sample
...     )(sample, fields=fields)

3. Fit the spectrum:

>>> from mars.optimization import SpectrumFitter
>>> fitter = SpectrumFitter(
...     x_exp=fields,
...     y_exp=y_exp,
...     param_space=param_space,
...     spectra_simulator=simulator,
...     norm_mode="integral"
... )
>>> result = fitter.fit(backend="optuna", n_trials=100)

4. Analyse the parameter space with `SpaceSearcher`:

>>> from mars.optimization import SpaceSearcher
>>> searcher = SpaceSearcher(loss_rel_tol=0.5, k_neighbors=3)
>>> local_minima = searcher(result)
>>> print(f"Found {len(local_minima)} diverse local minima.")

5. Fit multiple datasets simultaneously (composite fitter):

>>> # Assume fitter1 and fitter2 are SpectrumFitter instances
>>> from mars.optimization import SpectrumCompositeFitter
>>> composite = SpectrumCompositeFitter([fitter1, fitter2], weights=[0.7, 0.3])
>>> result = composite.fit(backend="nevergrad", budget=200)
"""


from .fitter import ParamSpec, FitResult, ParameterSpace, SpectrumFitter,\
        print_trial_results, CWSpectraSimulator, Spectrum2DFitter,\
        SpectrumCompositeFitter

from .searcher import SpaceSearcher
from .uncertanity_analyzer import UncertaintyAnalyzer
from .interactions import VaryInteraction, VaryDEInteraction, SampleVary, SampleUpdator
from .penalty_computations import RepulsivePenalty