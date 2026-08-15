Stationary Spectra Population
==============================

Overview
--------

The :class:`mars.population.populators.stationary.StationaryPopulator` computes the population-dependent contribution to transition intensities for continuous-wave (CW) EPR spectroscopy.
This populator calculates the population difference between upper and lower resonant levels, which determines the net absorption or emission intensity.

Theory
------

Population Difference
~~~~~~~~~~~~~~~~~~~~~

For stationary EPR experiments under thermal equilibrium or photoexcited conditions, the signal intensity is proportional to the difference in populations between the resonant energy levels:

.. math::

   \Delta n = n_{\text{lower}} - n_{\text{upper}}

where the populations depend on either thermal distribution or context-defined initial conditions.

Thermal Equilibrium Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When no explicit initial populations are provided, the system assumes thermal equilibrium at temperature T. Populations follow the Boltzmann distribution:

.. math::

   n_i = \frac{\exp(-E_i / k_B T)}{\sum_j \exp(-E_j / k_B T)}

where E_i are the eigenenergies of the spin Hamiltonian in the applied magnetic field.

Context-Defined Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a Context object (see :ref:`context-general-information`) provides initial populations (e.g., triplet sublevel polarization following photoexcitation), these populations are used instead of thermal values.
The Context-defined populations can be specified in any molecular basis (such as the zero-field splitting basis or xyz basis) and will be transformed to the field-dependent eigenbasis.

Basis Transformation
~~~~~~~~~~~~~~~~~~~~

If initial populations are defined in a non-eigenbasis, they are transformed using the eigenvectors of the full spin Hamiltonian:

.. math::

   n'_i = \sum_k |U_{ik}|^2 n_k

where U is the transformation matrix from the initial basis to the field-dependent eigenbasis, and :math:`|U_{ik}|^{2}` represents the squared overlap between basis states.

Implementation
--------------

The ``StationaryPopulator.forward()`` method:

1. Computes initial populations from temperature or Context
2. Transforms populations to the field-dependent eigenbasis if necessary
3. Calculates population differences between resonant levels (lvl_down, lvl_up)
4. Returns the population factor to be multiplied by transition matrix elements


Usage Notes
-----------

* If no Context is provided, thermal populations at ``init_temperature`` are used
* The ``full_system_vectors`` parameter is required only when populations need basis transformation
* This class does not handle magnetization or coherences - only diagonal density matrix elements (populations)
