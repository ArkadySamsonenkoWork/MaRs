"""
mars.population – Spin population dynamics and relaxation

This module provides tools for computing spin population evolution and spin-polarized spectra under
various physical regimes:
  • Level‑based kinetics (rate equations)
  • Density‑matrix evolution (Liouville space)
  • RWA (Rotating‑Wave Approximation) for driven systems
  • Lindblad / Redfield relaxation channels
  • Thermal and non‑thermal initial conditions

Main classes
------------
Context                – Container for spin system, rates, and basis
LevelBasedPopulator    – Rate‑equation kinetics on energy levels
RWADensityPopulator    – Density matrix under RWA
PropagatorDensityPopulator – Propagator‑based density evolution
StationaryPopulator    – Steady‑state / thermal equilibrium populations
LindbladRelaxationChannel – Lindblad master equation
RedfieldRelaxationChannel – Redfield relaxation theory
EvolutionSolver        – ODE solvers (exponential, Runge‑Kutta)

Example: setting up a population context for a powder sample
------------------------------------------------------------
>>> from mars import population, spin_model
>>>
>>> # Define spin system and sample (as in the main example)
>>> g_tensor = spin_model.Interaction(components=[2.0, 2.01, 2.02])
>>> zfs = spin_model.DEInteraction((100e6, 15e6))
>>> system = spin_model.SpinSystem(
...     electrons=[1.0],
...     g_tensors=[g_tensor],
...     electron_electron=[(0, 0, zfs)]
... )
>>> sample = spin_model.MultiOrientedSample(
...     base_spin_system=system,
...     gauss=5e-4
... )
>>>
>>> # Create a population context with custom initial populations
>>> context = population.Context(
...     sample=sample,
...     init_populations=[0.3, 0.6, 0.4],   # populations of the eigenstates
...     basis="xyz"                         # basis for the populations
... )
>>>
>>> # Now use this context in spectra_manager or population solvers
>>> from mars.spectra_manager import StationarySpectra
>>> manager = StationarySpectra(
...     freq=9.7e9,
...     context=context,
...     sample=sample,
...     harmonic=0,
... )
>>> # ... continue with simulation
"""


from .populators.core import BaseTimeDepPopulator, BasePopulator
from .contexts import BaseContext, Context, SummedContext, KroneckerContext, multiply_contexts
from .populators.stationary import StationaryPopulator, StationaryPopulatorExpanded
from .populators.level_population import LevelBasedPopulator, T1Populator
from .populators.density_population import RWADensityPopulator, PropagatorDensityPopulator
from .parametric_dependance import profiles, rates
from .concatination import concat_contexts

from .relaxation_channels.redfield import RedfieldRelaxationChannel
from .relaxation_channels.lindblad import LindbladRelaxationChannel
from .relaxation_channels.base_couling_channels import CouplingChannelManager

from .tr_utils import EvolutionPopulationSolver, EvolutionPropagatorSolver, EvolutionRWASolver
from . import matrix_generators, tr_utils, transform