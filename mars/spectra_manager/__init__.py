"""
mars.spectra_manager – EPR spectrum simulation and processing

This module computes EPR spectra from spin systems, supporting:
  • Continuous‑wave (CW) spectra – absorption or derivative mode
  • Time‑resolved spectra – pump‑probe, transient signals
  • Density‑matrix based spectra – full quantum evolution
  • Powder / single‑crystal averaging via orientation meshes
  • Integration with population dynamics for polarised spectra

Main classes
------------
StationarySpectra      – CW spectra (steady‑state)
CoupledTimeSpectra     – Time‑resolved spectra using coupled rate equations
DensityTimeSpectra     – Time‑resolved spectra from density‑matrix evolution

All spectral classes share a common call signature:
    manager(sample, fields) -> spectrum (torch.Tensor)

Examples
--------
1. Basic CW spectrum with a simple spin system:
>>> import torch
>>> import matplotlib.pyplot as plt
>>> from mars import spectra_manager, spin_model
>>>
>>> # Define a single electron with an anisotropic g‑tensor
>>> g_tensor = spin_model.Interaction(components=[2.0, 2.01, 2.02])
>>> system = spin_model.SpinSystem(electrons=[0.5], g_tensors=[g_tensor])
>>> sample = spin_model.MultiOrientedSample(
...     base_spin_system=system,
...     gauss=5e-4               # broadening (T)
... )
>>>
>>> # Create a stationary (CW) spectrum manager
>>> manager = spectra_manager.StationarySpectra(
...     freq=9.7e9,              # 9.7 GHz
...     sample=sample
... )
>>> fields = torch.linspace(0.34, 0.35, 1000)   # Tesla
>>> spectrum = manager(sample, fields=fields)
>>> plt.plot(fields, spectrum)
>>> plt.show()

2. CW spectrum with a population context (non‑thermal populations):
>>> from mars import population
>>>
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
>>> context = population.Context(
...     sample=sample,
...     init_populations=[0.3, 0.6, 0.4],
...     basis="xyz"
... )
>>>
>>> manager = spectra_manager.StationarySpectra(
...     freq=9.7e9,
...     context=context,
...     sample=sample,
...     harmonic=0                # absorption mode
... )
>>> fields = torch.linspace(0.335, 0.355, 1000)
>>> spectrum = manager(sample, fields=fields)

3. Time‑resolved spectra (coupled and density‑based):
>>> # Coupled (rate‑equation) time spectra
>>> kinetic = spectra_manager.CoupledTimeSpectra(
...     freq=9.7e9,
...     context=context,
...     sample=sample,
...     harmonic=0
... )
>>>
>>> # Density‑matrix time spectra
>>> density = spectra_manager.DensityTimeSpectra(
...     freq=9.7e9,
...     context=context,
...     sample=sample,
...     harmonic=0
... )
>>> # Both can be called with time points and field arrays
"""


from .spectra_manager import *
from .spectra_manager_expanded import StationarySpectraExpanded
from .wave_calculator import *
from .direct_managers import *

from .res_line_solvers import res_field_algorithm, res_freq_algorithm,\
    secular_approximation_algorithm, fixed_fields_algorithm
