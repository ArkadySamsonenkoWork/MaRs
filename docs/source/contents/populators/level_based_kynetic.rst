.. _level_based_kinetic_approach:

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

* **W**: Spontaneous (thermal) transition rates (thermal rates) modified to satisfy detailed balance
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
* **Thermal transitions (W)**: Spontaneous transitions satisfying detailed balance at temperature T
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

Constructing :math:`\exp(Kt)` directly, as a dense :math:`[T, \ldots, N, N]` tensor (one
full :math:`N \times N` matrix per time point), would be expensive in both memory and
time. ``stationary_rate_solver`` avoids this entirely by using the eigen-decomposition of
K to reduce the whole time-dependence to element-wise operations on vectors.

Derivation
^^^^^^^^^^^


Let

.. math::

   K = S J S^{-1}

where :math:`J = \mathrm{diag}(\lambda_1, \ldots, \lambda_N)` collects the eigenvalues of
K and the columns of S are the corresponding eigenvectors. Then

.. math::

   \exp(Kt) = S \exp(Jt)\, S^{-1}

Substituting into :math:`n(t) = \exp(Kt)\, n(0)` and introducing the coefficient vector

.. math::

   c \equiv S^{-1} n(0)

gives

.. math::

   n(t) = S \exp(Jt)\, c = \sum_{k=1}^{N} c_k\, e^{\lambda_k t}\, S_{:,k}

Only the difference of populations between the "down" and "up" level of each transition
is ever needed, so n(t) itself is never reconstructed. Defining the per-mode weight

.. math::

   w_k \equiv S_{\text{lvl\_down}, k} - S_{\text{lvl\_up}, k}

the observed quantity is

.. math::

   \Delta n(t) \;=\; n_{\text{lvl\_down}}(t) - n_{\text{lvl\_up}}(t)
              \;=\; \mathrm{Re}\left[\sum_{k=1}^{N} w_k\, c_k\, e^{\lambda_k t}\right]

The real part is extracted because the eigendecomposition of the kinetic matrix may involve complex eigenvalues and eigenvectors.
In exact arithmetic the population difference is real, but finite-precision arithmetic can leave a small residual imaginary component.

Aditionally, S⁻¹ is never explicitly formed. The coefficient vector :math:`c = S^{-1} n(0)` is
obtained by solving the linear system

.. math::

   S\, c = n(0)

with ``torch.linalg.solve(eig_vecs, n0)``. Solving the linear system (one LU factorization plus a
back-substitution, :math:`\mathcal{O}(N^3)`) is cheaper and numerically better behaved
than forming the inverse explicitly.

Order of operations.
^^^^^^^^^^^^^^^^^^^^

The solver exploits the constancy of the rate matrix :math:`K`.  
The population vector evolves as

.. math::

   n(t) = \exp(Kt)\, n(0).

To avoid constructing the dense matrix exponential :math:`\exp(Kt)` (which would cost
:math:`\mathcal{O}(N^3)` per time point and require :math:`\mathcal{O}(N^2)` storage),
the algorithm uses the eigen‑decomposition of :math:`K`:

.. math::

   K = S\, J\, S^{-1}, \qquad J = \mathrm{diag}(\lambda_1, \dots, \lambda_N),

where :math:`S` is the matrix of eigenvectors and :math:`\lambda_k` the eigenvalues.
Then

.. math::

   n(t) = S \exp(Jt)\, S^{-1} n(0).

Define :math:`c = S^{-1} n(0)`. The observable (population difference between the
lower and upper level of a transition) is

.. math::

   \Delta n(t) \;=\; n_{\mathrm{lvl\_down}}(t) - n_{\mathrm{lvl\_up}}(t)
   \;=\; \mathrm{Re}\left[\sum_{k=1}^N
           \bigl(S_{\mathrm{lvl\_down},k} - S_{\mathrm{lvl\_up},k}\bigr)\,
           c_k\, e^{\lambda_k t}\right].

The computation proceeds as follows.

1. **Build the rate matrix**

   Construct :math:`K` once (it is time-independent).

2. **Eigen-decomposition**

   Compute :math:`\lambda_k` and :math:`S` from :math:`K`.

   * Cost: :math:`\mathcal{O}(N^3)` — the dominant step.

   * Memory: :math:`\mathcal{O}(N^2)` for the eigenvector matrix.

3. **Solve for coefficients**

   Obtain :math:`c` by solving the linear system :math:`S\,c = n(0)`.
   Solving the system (one LU factorisation) costs :math:`\mathcal{O}(N^3)`, but is done
   only once.

   * Memory: :math:`\mathcal{O}(N)` for the coefficient vector.

4. **Evolve coefficients in time**

   For each time point :math:`t` in the requested grid, compute the vector
   :math:`e^{\lambda_k t}` for all :math:`k`. This is an element-wise operation of
   length :math:`N`, repeated for :math:`T` time points.

   * Cost: :math:`\mathcal{O}(T N)` in total.

   * Memory: a tensor of shape :math:`[T, \ldots, N]` for the exponential factors,
     i.e. :math:`\mathcal{O}(T N)` (not :math:`\mathcal{O}(T N^2)`).

5. **Assemble the observable**

   Compute the weight vector
   :math:`w_k = S_{\mathrm{lvl\_down},k} - S_{\mathrm{lvl\_up},k}`
   (length :math:`N`). Then form the dot product with the time-evolved coefficients
   and take the real part to obtain :math:`\Delta n(t)` for each time point.

   * Cost: :math:`\mathcal{O}(T N)` (one dot product per time point).

   * Memory: only the output signal of length :math:`T`.

Steps 2-4 are independent of the time grid and are performed once.

Quasi-Stationary Solution
~~~~~~~~~~~~~~~~~~~~~~~~~

When K depends on time but not on populations, the solution can be computed iteratively
(``EvolutionPopulationSolver.exponential_solver``):

.. math::

   n(t_{i+1}) = \exp(K(t_i) \Delta t) \cdot n(t_i)

The matrix exponentials for all intervals are precomputed and batched together
(``torch.matrix_exp(M[:-1] * dt)``, shape ``[T-1, ..., N, N]``), rather than being
recomputed one interval at a time. This is a genuine time/memory
trade-off relative to ``odeint_solver``: it increases memory overhead (a full
:math:`N \times N` matrix per time step is stored) but reduces total computation time,
since a batched call to ``torch.matrix_exp`` replaces the many small adaptive
Runge-Kutta steps that ``odeint_solver`` would otherwise take.

This piecewise-constant approximation is only valid when K changes slowly compared to the
relaxation process it drives - the underlying assumption is that K can be treated as
constant over a single interval :math:`[t_i, t_{i+1}]`. Quantitatively, the relative rate
of change of K over one relaxation time must be small:

.. math::

   \frac{1}{\lVert K \rVert}\left\lVert \frac{dK}{dt} \right\rVert
   \;\ll\; \left|\mathrm{Re}(\lambda_{\min})\right|

where :math:`\lambda_{\min}` is the eigenvalue of K with the smallest magnitude of its
real part, i.e. the slowest relaxation channel. In words: within one relaxation time,
:math:`1/\left|\mathrm{Re}(\lambda_{\min})\right|`, K must not change appreciably. If this
condition is not satisfied - K varies on a timescale comparable to or faster than
relaxation - the piecewise-stationary approximation breaks down and ``odeint_solver``
(or a finer time grid) should be used instead.

Adaptive ODE Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

For the general case where K = K(n, t), the equation is solved using adaptive Runge-Kutta methods (via ``torchdiffeq``). This provides automatic time-step control but is computationally more expensive.


The solver is automatically selected based on the Context:

* **Stationary**: K constant → matrix exponential

* **Time-dependent**: K(t) → adaptive ODE solver by default

Diagonalizability of the Kinetic Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The eigen-decomposition used by ``stationary_rate_solver`` (see
:ref:`diagonalizability_order` above) requires K to be diagonalizable, i.e. to have a
complete set of N linearly independent eigenvectors so that S is invertible. This is not
guaranteed for an arbitrary matrix, but it does hold in the two cases relevant to this
library:

* Symmetric matrices are always diagonalizable. By the spectral theorem, any real
  symmetric matrix has a complete set of orthogonal eigenvectors and real eigenvalues,
  so S is not merely invertible but can be chosen orthogonal
  (:math:`S^{-1} = S^{T}`).

* A thermalized rate matrix is diagonalizable. The thermal part W of K is constructed
  to satisfy detailed balance with respect to the Boltzmann populations
  :math:`p_i^{\text{eq}} \propto \exp(-E_i / k_B T)`:

  .. math::

     \frac{w_{ij}}{w_{ji}} = \exp\left(-\frac{E_i - E_j}{k_B T}\right)
                            = \frac{p_i^{\text{eq}}}{p_j^{\text{eq}}}

  Define :math:`D = \mathrm{diag}\!\left(\sqrt{p_1^{\text{eq}}}, \ldots, \sqrt{p_N^{\text{eq}}}\right)`.
  A direct substitution shows that :math:`\hat{W} \equiv D^{-1} W D` is symmetric.

If K is defective (not diagonalizable), :math:`\mathrm{cond}(S) \to \infty`, the
eigen-based solution can lose all significant digits even though the underlying floating
point arithmetic itself is exact. Physically this also reflects a real property of the
kinetic model - tiny changes in the rate constants can strongly change relaxation times
and mode amplitudes, so the model itself is structurally sensitive to the
(experimentally uncertain) rate parameters. In this regime, use
``EvolutionPopulationSolver.stationary_rate_solver_expm``, which computes
:math:`\exp(Kt)` directly (via matrix exponentiation, without eigen-decomposition) and
remains mathematically valid for defective matrices, at the cost of being slower.

Usage Examples
--------------

Both classes referenced in this document can be imported from ``mars.population``:

.. code-block:: python

   from mars.population import tr_utils
   from mars.population import LevelBasedPopulator

Default solver
~~~~~~~~~~~~~~

If ``solver`` is not given (``solver=None``), ``LevelBasedPopulator`` picks it
automatically from the Context:

* ``EvolutionPopulationSolver.odeint_solver`` if the relaxation parameters are
  time-dependent,
* ``EvolutionPopulationSolver.stationary_rate_solver`` (the eigen-decomposition solver
  described above) if K is constant in time.

.. code-block:: python

   # solver is chosen automatically based on the Context
   populator = LevelBasedPopulator(
       context=context,
       init_temperature=293.0,
   )

Passing an arbitrary solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any callable matching the ``EvolutionSolver`` signature
``solver(time, initial_populations, evo, matrix_generator, lvl_down, lvl_up)`` can be
passed explicitly, overriding the automatic choice - for example, to force the robust
matrix-exponential solver when K is expected to be close to defective:

.. code-block:: python

   populator = level_population.LevelBasedPopulator(
       context=context,
       solver=tr_utils.EvolutionPopulationSolver.stationary_rate_solver_expm,
       init_temperature=293.0,
   )

Using the quasi-stationary solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When K(t) changes with time, but slowly compared to the relaxation it drives (see the
condition in :ref:`quasi_stationary_solution`), ``exponential_solver`` can be used instead
of the adaptive ODE integrator:

.. code-block:: python

   populator = LevelBasedPopulator(
       context=context,
       solver=tr_utils.EvolutionPopulationSolver.exponential_solver,
       init_temperature=293.0,
   )

Limitations
-----------

This approach: Treats only populations (diagonal density matrix elements)
For systems where coherences are important, use the density matrix approaches (RWA or propagator computation method).