"""
MaRs — Magnetic Resonance Simulations (PyTorch‑based EPR library)

MaRs provides a complete framework for simulating EPR spectra, spin dynamics,
and parameter optimisation. It is built on PyTorch for GPU acceleration and
automatic differentiation.

Subpackages
-----------
population      – Spin population dynamics and relaxation
spectra_manager – Spectrum simulation (CW, time‑resolved, density‑matrix)
mesher          – Orientation meshing for powder / single‑crystal averaging
optimization    – Parameter fitting against experimental data
visualization   – Plotting and analysis tools

Quick Example (using all major components)
------------------------------------------
>>> import torch
>>> import matplotlib.pyplot as plt
>>> from mars import spectra_manager, spin_model, population
>>>
>>> # 1. Define a spin system
>>> g_tensor = spin_model.Interaction(components=[2.0, 2.01, 2.02])
>>> zfs = spin_model.DEInteraction((100e6, 15e6))   # zero‑field splitting
>>> system = spin_model.SpinSystem(
...     electrons=[1.0],
...     g_tensors=[g_tensor],
...     electron_electron=[(0, 0, zfs)]
... )
>>>
>>> # 2. Create a powder sample (orientation averaging)
>>> sample = spin_model.MultiOrientedSample(
...     base_spin_system=system,
...     gauss=5e-4               # Gaussian broadening (T)
... )
>>>
>>> # 3. Choose a spectrum manager
>>> manager = spectra_manager.StationarySpectra(
...     freq=9.7e9,              # 9.7 GHz
...     sample=sample,
... )
>>>
>>> # 4. Simulate over a field range
>>> fields = torch.linspace(0.335, 0.355, 1000)   # Tesla
>>> spectrum = manager(sample, fields=fields)
>>>
>>> # 6. Plot
>>> plt.plot(fields, spectrum)
>>> plt.show()
"""


from .serialization import serialization, graph_representation, operations_interface
from .operations import concat, flatten, stack, expand, repeat, unsqueeze, squeeze, transpose, mask
from .multiplication import multiply
from .save_procedures.general_procedures import save, load
from .reader import read_bruker_data

from . import constants
from . import spectra_manager
from . import spin_model
from . import population
from . import spectra_processing
from . import mesher
from . import visualization
