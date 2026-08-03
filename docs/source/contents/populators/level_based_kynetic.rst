Level-Based Kinetic Approach
=============================

Overview
--------

The :class:`mars.population.level_population.LevelBasedPopulator` class implements time-resolved EPR signal modeling using the kinetic (population-based) relaxation paradigm.
This approach computes evolution of only the diagonal elements of the density matrix - the populations of energy levels and models their evolution through rate equations.

Theory
------

Kinetic Equation
~~~~~~~~~~~~~~~~

The time evolution of level populations is governed by the kinetic equation:

.. math::

   \frac{dn}{dt} = K(t, n) \cdot n

where:

* **n** is the vector of populations of all energy levels
* **K** is the kinetic matrix encoding transition rates between levels

The kinetic matrix incorporates three types of processes:

.. math::

   K = W + D - \text{diag}(O)

where:

* **W**: Spontaneous (thermal) transition rates (free probabilities) modified to satisfy detailed balance
* **D**: Driven (induced) transition rates from external perturbations
* **\text{diag}(O)**: Population outgoing rates (e.g., phosphorescence decay from triplet states)


When relaxation coupling channels (Redfield or Lindblad) are provided via the
``relaxation_coupling_channels`` argument of :class:`~mars.population.contexts.Context`,
their population transfer contributions are automatically extracted and added to the
kinetic matrix.  For each channel, the block of the Redfield tensor that describes
population flow (the :math:`R_{aacc}` components) is computed and summed into **K**.
See :ref:`relaxation_channels` for details on how to define such channels.


Signal Intensity
~~~~~~~~~~~~~~~~

The time-dependent EPR signal is proportional to:

.. math::

   I(t) \propto \Delta n(t) |M|^2 = [n_{\text{lower}}(t) - n_{\text{upper}}(t)] |M|^2

where M is the transition matrix element for transverse magnetization.

Initial Conditions
~~~~~~~~~~~~~~~~~~

Initial populations are determined by:

1. **Thermal equilibrium** at temperature T (Boltzmann distribution)
2. **Context-defined** populations (e.g., triplet mechanism with selective population)

Populations defined in a molecular basis are transformed to the field-dependent eigenbasis using the squared overlap matrix :math:`|U|^{2}`.

Relaxation Mechanisms
~~~~~~~~~~~~~~~~~~~~~

The Context object (see :ref:`relaxation_parameters` for more inforamtion) encodes physical relaxation processes:

* **Losses (O)**: Depopulation without transitions to other spin states (e.g. low singlet state)
* **Free transitions (W)**: Spontaneous transitions satisfying detailed balance at temperature T
* **Induced transitions (D)**: Externally driven transitions (e.g., by microwave field)

To carry out a detailed balance, "Mars" forces (see :ref:`detailed_balance`):

.. math::

   \frac{W_{ij}}{W_{ji}} = \exp\left(\frac{E_j - E_i}{k_B T}\right)

Time-Dependent Relaxation
~~~~~~~~~~~~~~~~~~~~~~~~~

The MaRs library supports relaxation parameters that depend on time, enabling modeling of systems where macroscopic properties (e.g., temperature) change during evolution.
In this case, K becomes K(t).

Numerical Solutions
-------------------

Stationary Solution
~~~~~~~~~~~~~~~~~~~

When K is independent of time and populations, the evolution has a closed-form solution:

.. math::

   n(t) = \exp(Kt) \cdot n(0)

This is computed efficiently via matrix diagonalization:

.. math::

   \exp(Kt) = S \exp(Jt) S^{-1}

where J is the diagonal eigenvalue matrix of K and S contains its eigenvectors.
In this approach it is needed to compute diagonal of exp(Jt) which has the shape [N] (not [N, N] as it was in the case of computation matrix exponential).
Moreover, in this approach via batch computation the creator of matrices with the shape [t, ..., N, N] is not needed anymore.

Quasi-Stationary Solution
~~~~~~~~~~~~~~~~~~~~~~~~~

When K depends on time but not on populations, the solution can be computed iteratively:

.. math::

   n(t_{i+1}) = \exp(K(t_i) \Delta t) \cdot n(t_i)

The matrix exponential is precomputed at each time step.

Adaptive ODE Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

For the general case where K = K(n, t), the equation is solved using adaptive Runge-Kutta methods (via ``torchdiffeq``). This provides automatic time-step control but is computationally more expensive.


The solver is automatically selected based on the Context:
* **Stationary**: K constant → matrix exponential
* **Time-dependent**: K(t) → adaptive ODE solver by default

Limitations
-----------

This approach:
* Treats only populations (diagonal density matrix elements)
For systems where coherences are important, use the density matrix approaches (RWA or propagator computation method).