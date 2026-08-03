.. _fitting_tutorial:

Fitting Spectroscopic Data
==========================

MaRs provides a flexible framework for fitting experimental spectroscopic data using modern optimization libraries such as Optuna and Nevergrad.

Defining the Parameter Space
----------------------------

All fitting procedures start by defining which parameters are allowed to vary and within what bounds.
This is done using the :class:`mars.optimization.fitter.ParameterSpace` and :class:`mars.optimization.fitter.ParamSpec`.

.. code-block:: python

    from mars.optimization import ParameterSpace, ParamSpec

    param_specs = [
        ParamSpec(name="g", bounds=(2.0, 2.3), default=2.05),
        ParamSpec(name="D",  bounds=(100e6, 1000e6), default=200e6),
        ParamSpec(name="J",  bounds=(5e9, 40e9),   default=10e6),
        ParamSpec(name="ham_strain", bounds=(2e7, 1e8), default=5e7),
        ParamSpec(name="lorentz", bounds=(0.0, 1e-3), default=1e-4)
    ]

    param_space = ParameterSpace(
        specs=param_specs,
        fixed_params={"temperature": 10.0, "freq": 34e9}
    )

Creating a Spectra Simulator
----------------------------

The optimizer needs a callable that maps a parameter dictionary to a simulated spectrum:

.. code-block:: python

    class CWSpectraSimulator:
        def __init__(self, init_params):
            sample = create_sample(init_params)  # user-defined function
            self.spectra_creator = StationarySpectra(
                freq=init_params["freq"],
                sample=sample,
                temperature=init_params["temperature"]
            )

        def __call__(self, fields, params):
            sample = create_sample(params)  # user-defined function
            return self.spectra_creator(sample, fields)

For cases where only context-dependent quantities (e.g., populations or kinetic rates) change while spin Hamiltonian parameters remain fixed, you can cache resonance data to speed up evaluation:

.. code-block:: python

    class CWSpectraSimulator:
        def __init__(self, init_params):
            self.sample = create_sample(init_params)  # user-defined function
            self.spectra_creator = StationarySpectra(
                recompute_spin_parameters=False,  # resonance positions/vectors won't be recomputed
                freq=init_params["freq"],
                sample=self.sample,
                temperature=init_params["temperature"]
            )

        def __call__(self, fields, params):
            context = create_context(params)  # user-defined function
            self.spectra_creator.update_context(context)  # update context to new polarization parameters
            return self.spectra_creator(self.sample, fields)

Objective Functions
-------------------

MaRs supports multiple objective functions for comparing simulated and experimental spectra.
All objectives inherit from :class:`mars.optimization.objectives.BaseObjectiveFunction` and return a scalar loss.

.. code-block:: python

    from mars.optimization.objectives import MSEObjective, CosineSimilarity

    fitter = SpectrumFitter(
        x_exp=fields_exp,
        y_exp=spectrum_exp,
        param_space=param_space,
        spectra_simulator=CWSpectraSimulator(),
        norm_mode="max",
        objective=MSEObjective()  # or CosineSimilarity(), etc.
    )

Fitting with Optuna Samplers
----------------------------

MaRs supports two optimization libraries for parameter fitting: *Optuna* and *Nevergrad*. While both are powerful, they differ in design philosophy and ease of use:

- **Optuna** provides a smaller but well curated set of samplers that are easy to configure and highly effective for most spectroscopic fitting tasks.

- **Nevergrad** offers a much broader collection of optimization algorithms - from gradient-free methods to evolutionary and population-based strategies,
making it highly flexible but also more challenging to navigate without prior experience.

Optuna samplers (e.g., ``TPESampler``, ``CmaEsSampler``, ``NSGAIISampler``) offers several samplers suitable for different stages of optimization.

**Tree-structured Parzen Estimator (TPE)**

.. code-block:: python

    import optuna
    from mars.optimization import SpectrumFitter

    fitter = SpectrumFitter(
        x_exp=fields_exp,
        y_exp=spectrum_exp,
        param_space=param_space,
        spectra_simulator=CWSpectraSimulator(),
        norm_mode="max"
    )

    sampler_tpe = optuna.samplers.TPESampler(
        seed=42,
        multivariate=True,
        n_startup_trials=50
    )

    result_tpe = fitter.fit(
        backend="optuna",
        n_trials=300,
        sampler=sampler_tpe,
        study_name="tpe_fit"
    )

**Covariance Matrix Adaptation Evolution Strategy (CMA-ES)**

.. code-block:: python

    sampler_cmaes = optuna.samplers.CmaEsSampler(
        seed=42,
        n_startup_trials=30,
        restart_strategy="ipop"
    )

    result_cmaes = fitter.fit(
        backend="optuna",
        n_trials=300,
        sampler=sampler_cmaes,
        study_name="cmaes_fit"
    )

**Bayesian Optimization with BoTorch**

.. code-block:: python

    sampler_botorch = optuna.integration.BoTorchSampler(
        seed=42,
        n_startup_trials=20
    )

    result_botorch = fitter.fit(
        backend="optuna",
        n_trials=100,
        sampler=sampler_botorch,
        study_name="botorch_fit"
    )

In general, MaRs can accept any additional arguments and pass them to ``optuna.create_study()``. For more possibilities, see the `Optuna documentation <https://optuna.readthedocs.io/en/stable/>`_.

Optuna Dashboard
----------------

MaRs supports launching the Optuna dashboard during fitting for real-time monitoring. Set ``run_dashboard=True`` in the ``fit()`` call:

.. code-block:: python

    result = fitter.fit(backend="optuna", n_trials=300, run_dashboard=True)

This starts a local web server (typically at http://localhost:8080) where you can inspect trial history, parameter importance, and convergence behavior. The dashboard requires the ``optuna-dashboard`` package to be installed.

Fitting with Nevergrad
----------------------

`Nevergrad <https://github.com/facebookresearch/nevergrad>` is a gradient-free optimization library. It provides a unified interface to more than 100 derivative-free optimization algorithms, including evolutionary strategies (e.g., CMA-ES), Bayesian optimization variants, particle swarm methods, and classical direct search techniques like COBYLA and Powell.

Example: using the COBYLA optimizer (a constrained optimization algorithm based on linear approximations):

.. code-block:: python

    result_cobyla = fitter.fit(
        backend="ng",
        budget=300,          # maximum number of function evaluations
        optimizer="Cobyla"
    )

The full list of optimizers available through Nevergrad in MaRs can be inspected at runtime:

.. code-block:: python

    print(SpectrumFitter.__available_optimizers__["nevergrad"])

This includes popular choices such as:
- ``"CMA"`` (Covariance Matrix Adaptation Evolution Strategy) — robust for continuous, non-convex problems,
- ``"DE"`` (Differential Evolution) — effective for multimodal landscapes,
- ``"PSO"`` (Particle Swarm Optimization),
- ``"OnePlusOne"`` — simple yet surprisingly effective for low-dimensional problems,
- and many more.

Weighted Multi-Spectrum Fitting
-------------------------------

When fitting multiple spectra simultaneously, you can assign different weights to each dataset using the ``weights`` argument. This is useful when some spectra are noisier or more critical than others:

.. code-block:: python

    fitter = SpectrumFitter(
        x_exp=[fields1, fields2],
        y_exp=[spectrum1, spectrum2],
        param_space=param_space,
        spectra_simulator=MultiSpectraSimulator(),
        weights=[1.0, 0.5],  # second spectrum contributes half as much to total loss
        norm_mode="max"
    )

2D Spectral Fitting with Spectrum2DFitter
-----------------------------------------

For time-resolved or other 2D spectroscopic data, it is possible to use :class:`mars.optimization.fitter.Spectrum2DFitter`. It accepts two independent axes (e.g., magnetic field and time) and a 2D intensity array.

The simulator must accept three arguments: ``x1``, ``x2``, and ``params``.

Example setup:

.. code-block:: python

    class TRSpectraSimulator:
        def __call__(self, fields, times, params):
            # fields: 1D tensor (magnetic field axis)
            # times: 1D tensor (time axis)
            # returns: 2D tensor of shape (len(times), len(fields))
            sample = create_sample(params)
            context = create_context(params)
            tr_creator = TimeResolvedSpectra(
                context=context,
                freq=params["freq"],
                sample=sample,
                temperature=params["temperature"]
            )
            return tr_creator(sample, fields, times)

    fitter_2d = Spectrum2DFitter(
        x1_exp=fields,      # magnetic field (T)
        x2_exp=times,       # time (s)
        y_exp=spectrum_2d,  # 2D experimental data
        param_space=param_space,
        spectra_simulator=TRSpectraSimulator(),
        norm_mode="integral"
    )

    result_2d = fitter_2d.fit(backend="optuna", n_trials=200)

Like its 1D counterpart, :class:`Spectrum2DFitter` supports multi-dataset fitting (list inputs), custom objectives, and weighted losses.

Composite Multi-Fitter
----------------------

When fitting several datasets that share the same parameter space but use different simulators or objective functions,
use :class:`mars.optimization.fitter.SpectrumCompositeFitter`.
It aggregates individual :class:`BaseSpectrumFitter` instances and minimizes a weighted sum of their losses.

.. code-block:: python

    from mars.optimization import SpectrumCompositeFitter

    fitter_cw = SpectrumFitter(
        x_exp=fields_cw,
        y_exp=spectrum_cw,
        param_space=param_space,
        spectra_simulator=CWSpectraSimulator(),
        norm_mode="max"
    )

    fitter_tr = Spectrum2DFitter(
        x1_exp=fields_tr,
        x2_exp=times,
        y_exp=spectrum_2d,
        param_space=param_space,
        spectra_simulator=TRSpectraSimulator(),
        norm_mode="integral"
    )

    composite = SpectrumCompositeFitter(
        fitters=[fitter_cw, fitter_tr],
        weights=[1.0, 0.5]   # second fitter contributes half as much
    )

    result = composite.fit(backend="optuna", n_trials=300)

All sub-fitters must share an identical :class:`ParameterSpace`. The ``fit()`` interface is the same as for :class:`SpectrumFitter`, supporting both Optuna and Nevergrad backends.

Penalty-Based Optimization
--------------------------

For landscapes with multiple local minima, the penalty variant of the optimizer discourages repeated convergence to already-found solutions.
The penalty term is updated periodically and the optimizer is warm-restarted with recomputed penalized losses.
Final results always report raw (unpenalized) losses.

.. code-block:: python

    result = fitter.fit(
        backend="optuna",
        n_trials=300,
        use_penalty=True,
        penalty_names=["g", "D"],   # dimensions along which to penalize; defaults to all
        update_penalty_every=20,    # recompute penalty state every N trials
        restart_every=60,           # warm-restart the study every N trials
        penalty_force=1.0           # penalty strength multiplier
    )

``use_penalty=True`` is supported by both Optuna and Nevergrad backends on :class:`SpectrumFitter` and :class:`Spectrum2DFitter`.

Exploring Alternative Minima with SpaceSearcher
-----------------------------------------------

In complex landscapes, multiple parameter sets may yield similarly good fits.
The :class:`mars.optimization.fitter.SpaceSearcher` class identifies such alternatives that are distant from the best solution in parameter space.

.. code-block:: python

    from mars.optimization import SpaceSearcher

    searcher = SpaceSearcher(
        loss_rel_tol=0.2,      # accept trials with loss ≤ 1.2 × best_loss
        distance_fraction=0.2  # require minimum scaled Euclidean distance
    )

    alternatives = searcher(fit_result=result_tpe, param_names=["g", "D", "J"])
    print_trial_results(alternatives)

:class:`SpaceSearcher` works with results from both Optuna and Nevergrad backends and scales parameters before computing distances to ensure fair comparison across units.

For convenience, use :func:`mars.optimization.fitter.print_trial_results` to display results in a readable format:

.. code-block:: python

    from mars.optimization import print_trial_results
    print_trial_results(alternatives, max_params=6, precision=5)


Plotting Capabilities
*********************

In addition to manually simulating spectra, :class:`SpaceSearcher` provides several built-in plotting methods
for analysing the distribution of good parameters:

- :meth:`~mars.optimization.fitter.SpaceSearcher.plot_1d_marginals`:
  Weighted 1D histograms (marginal distributions) for each parameter, with the best-fit value marked.

- :meth:`~mars.optimization.fitter.SpaceSearcher.plot_2d_pairs`:
  Weighted 2D scatter plots for selected parameter pairs

- :meth:`~mars.optimization.fitter.SpaceSearcher.plot_clusters`:
  PCA projection of the parameter space coloured by HDBSCAN clusters.

- :meth:`~mars.optimization.fitter.SpaceSearcher.plot_loss_distribution`:
  Histogram of the loss values among the "good" trials.

- :meth:`~mars.optimization.fitter.SpaceSearcher.plot_correlation_matrix`:
  Correlation matrix of the parameters for the good trials.

- :meth:`~mars.optimization.fitter.SpaceSearcher.plot_parallel_coordinates`:
  Parallel-coordinates plot for the top-K trials.

- :meth:`~mars.optimization.fitter.SpaceSearcher.plot_rescaled_loss_scatter`:
  Scatter plots of each parameter versus a rescaled loss (min→0, avg→0.5, max→1), helping to identify parameter ranges that produce low losses.

- :meth:`~mars.optimization.fitter.SpaceSearcher.plot_loss_boxplots_by_bin`:
  Box-plots of the loss distribution across quantile-based bins of each parameter, highlighting regions of consistently good fits.

All these methods accept a ``show`` parameter (default ``True``) to display the figures immediately, and return the figure objects for further customisation.


Uncertainty Analysis
--------------------

After fitting, :class:`mars.optimization.fitter.UncertaintyAnalyzer` computes confidence intervals for the fitted parameters. Five methods are available:

- ``"hessian"`` — fast symmetric intervals via quadratic approximation of the loss surface.
- ``"profile"`` — asymmetric intervals via re-optimisation of nuisance parameters.
- ``"mcmc"`` — Bayesian credible intervals via ensemble sampling (requires ``emcee``).
- ``"trials"`` — bounding-box from existing optimiser trial history; no additional evaluations.
- ``"bootstrap"`` — distribution-free intervals via residual resampling; valid for any smooth loss.

The first three methods have correct statistical meaning only when the loss is proportional to SSE or MSE (the default in MaRs).
For custom or composite losses, use ``"bootstrap"``.

.. code-block:: python

    from mars.optimization import UncertaintyAnalyzer

    analyzer = UncertaintyAnalyzer(
        fitter=fitter,
        fit_result=result,
        method="profile",        # or "hessian", "mcmc", "trials", "bootstrap"
        confidence_level=0.95
    )

    uncertainty = analyzer(param_names=["g", "D", "J"])

    for name, ci in uncertainty.intervals.items():
        print(f"{name}: {ci.best:.4g}  [{ci.lower:.4g}, {ci.upper:.4g}]")

For quick diagnostics, ``"hessian"`` is recommended. For publication-quality asymmetric intervals, use ``"profile"`` with a finer grid (``n_points=20`` or more). The goodness-of-fit can be assessed independently via:

.. code-block:: python

    chi2 = analyzer.get_chi_square(return_reduced=True)
    print(f"Reduced chi-squared: {chi2:.3f}")   # ~1.0 indicates a good fit