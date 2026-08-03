Coupled Time-Resolved Spectra (Population Dynamics)
===================================================

The :class:`mars.spectra_manager.CoupledTimeSpectra` class computes time-resolved EPR spectra based on **population kinetics**. It is based on the solution of the rate equation:

.. math::

   \frac{d\mathbf{n}}{dt} = \mathbf{K} \mathbf{n}(t)

where :math:`\mathbf{n}` is the vector of level populations and :math:`\mathbf{K}` is a kinetic matrix constructed from user-defined relaxation pathways (thermal, out, driven transitions).

This approach captures spin polarization buildup and decay but **neglects quantum coherences** and off-diagonal density matrix elements.

Unlike :class:`TruncTimeSpectra`, this class computes the **full eigensystem** at each field point.

Key Capabilities
----------------

- Support for arbitrary initial populations via :class:`mars.population.contexts.Context`.
- Time-dependent spectra as 2D arrays (field × time).
- Compatible with all interaction types and sample configurations.

Examples
--------

**Example 1: Triplet decay with initial polarization**

.. code-block:: python

   # Define spin system (S=1)
   g = spin_model.Interaction(2.002)
   zfs = spin_model.DEInteraction([500e6, 100e6])
   sys = spin_model.SpinSystem(electrons=[1.0], g_tensors=[g], electron_electron=[(0,0,zfs)])
   sample = spin_model.MultiOrientedSample(sys, lorentz=0.001, ham_strain=1e7)

   # Context with initial pop [0.5, 0.3, 0.2] and relaxation rates
   ctx = population.Context(
       sample=sample, basis="eigen",
       init_populations=[0.5, 0.3, 0.2],
       decay_rates=torch.tensor([100., 100., 100.]),      # 10 ms depopulation
       thermal_rates=torch.tensor([[0, 1e3, 0], [1e3, 0, 1e3], [0, 1e3, 0]])  # 1 ms equilibration
   )

   tr_creator = spectra_manager.CoupledTimeSpectra(freq=9.8e9, sample=sample, context=ctx)
   fields = torch.linspace(0.30, 0.40, 500)
   time = torch.linspace(0, 0.02, 200)  # 0–20 ms
   spec_2d = tr_creator(sample, fields, time)  # shape: (500, 200)