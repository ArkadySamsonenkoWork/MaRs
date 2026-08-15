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

Constructing :math:`\exp(Kt)` directly, as a dense :math:`[T, \ldots, N, N]` tensor (one
full :math:`N \times N` matrix per time point), would be expensive in both memory and
time. ``stationary_rate_solver`` avoids this entirely by using the eigen-decomposition of
K to reduce the whole time-dependence to element-wise operations on vectors.

**Derivation**

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

S⁻¹ is never explicitly formed. The coefficient vector :math:`c = S^{-1} n(0)` is
obtained by solving the linear system

.. math::

   S\, c = n(0)

with ``torch.linalg.solve(eig_vecs, n0)``, instead of computing :math:`S^{-1}` and then
multiplying by it. Solving the linear system (one LU factorization plus a
back-substitution, :math:`\mathcal{O}(N^3)`) is cheaper and numerically better behaved
than forming the inverse explicitly - and it is the only thing actually needed, since the
inverse matrix itself is never used anywhere else in the computation.

**No [T, ..., N, N] matrix is ever created.** Because :math:`\exp(Jt)` is diagonal, only
its diagonal has to be evaluated. For a batch of time points this is a tensor of shape
``[T, ..., N]`` (the eigenvalues broadcast against time), never the full
``[T, ..., N, N]`` matrix that a direct evaluation of :math:`\exp(Kt)` would require. This
is the source of the speed/memory advantage of this solver: evaluating
``exp_factors = exp(t * eig_vals)`` and multiplying it element-wise by ``c`` costs
:math:`\mathcal{O}(N)` per time point, versus :math:`\mathcal{O}(N^3)` for a dense matrix
exponential per time point.

.. _diagonalizability_order:

**Order of operations.** The implementation of
``EvolutionPopulationSolver.stationary_rate_solver`` follows this order, and the order
matters:

1. Build ``M = K`` at ``time[0]`` (K is constant in time here, so a single evaluation
   suffices).
2. ``eig_vals, eig_vecs = torch.linalg.eig(M)`` - a single eigen-decomposition, shared by
   every time point. ``eig_vals`` has shape ``[..., N]``, ``eig_vecs`` (= S) has shape
   ``[..., N, N]``.
3. Check the conditioning of ``eig_vecs`` (see :ref:`diagonalizability` below) before
   trusting the decomposition.
4. ``c = torch.linalg.solve(eig_vecs, n0)`` - the coefficients, obtained without forming
   :math:`S^{-1}`.
5. Broadcast ``eig_vals`` and ``eig_vecs`` to the batch shape of ``c`` if they don't
   already match it.
6. Reshape ``time`` so that it lines up with the trailing eigenmode axis of ``eig_vals``
   for broadcasting. This has to happen *before* the exponential is taken, and the number
   of singleton dimensions inserted must match the number of batch dimensions of M -
   otherwise time ends up broadcast against the wrong axis and the result is silently
   wrong rather than raising an error.
7. ``exp_factors = exp(time * eig_vals)`` - shape ``[T, ..., N]``: only the diagonal
   values, never a full matrix.
8. Multiply in place by the coefficients: ``exp_factors *= c`` (fusing what corresponds to
   step 4 and step 7 into the same tensor, avoiding an extra allocation).
9. Compute the per-mode weight vectors, ``eig_vecs[..., lvl_down, :] - eig_vecs[..., lvl_up, :]``.
10. Multiply by ``exp_factors``, sum over the eigenmode axis, and take the real part.

Steps 2-4 do not depend on time, so they are computed once and reused for every time
point, rather than being recomputed inside a loop over time - this is what makes the
stationary solver much cheaper than repeatedly evaluating a matrix exponential.

.. _quasi_stationary_solution:

Quasi-Stationary Solution
~~~~~~~~~~~~~~~~~~~~~~~~~

When K depends on time but not on populations, the solution can be computed iteratively
(``EvolutionPopulationSolver.exponential_solver``):

.. math::

   n(t_{i+1}) = \exp(K(t_i) \Delta t) \cdot n(t_i)

The matrix exponentials for all intervals are precomputed and batched together
(``torch.matrix_exp(M[:-1] * dt)``, shape ``[T-1, ..., N, N]``), rather than being
recomputed one interval at a time - this is what makes the method fast, at the price of
holding all of these matrices in memory simultaneously. This is a genuine time/memory
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

.. _diagonalizability:

Diagonalizability of the Kinetic Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The eigen-decomposition used by ``stationary_rate_solver`` (see
:ref:`diagonalizability_order` above) requires K to be diagonalizable, i.e. to have a
complete set of N linearly independent eigenvectors so that S is invertible. This is not
guaranteed for an arbitrary matrix, but it does hold in the two cases relevant to this
library.

**Symmetric matrices are always diagonalizable.** By the spectral theorem, any real
symmetric matrix has a complete set of orthogonal eigenvectors and real eigenvalues, so S
is not merely invertible but can be chosen orthogonal (:math:`S^{-1} = S^{T}`).

**A thermalized rate matrix is diagonalizable.** The thermal part W of K is constructed
to satisfy detailed balance with respect to the Boltzmann populations
:math:`p_i^{\text{eq}} \propto \exp(-E_i / k_B T)`:

.. math::

   \frac{W_{ij}}{W_{ji}} = \exp\left(-\frac{E_i - E_j}{k_B T}\right)
                          = \frac{p_i^{\text{eq}}}{p_j^{\text{eq}}}

Define :math:`D = \mathrm{diag}\!\left(\sqrt{p_1^{\text{eq}}}, \ldots, \sqrt{p_N^{\text{eq}}}\right)`.
A direct substitution shows that :math:`\hat{W} \equiv D^{-1} W D` is symmetric:

.. math::

   \hat{W}_{ij} = W_{ij}\,\frac{\sqrt{p_j^{\text{eq}}}}{\sqrt{p_i^{\text{eq}}}}
   \qquad\Longrightarrow\qquad
   \hat{W}_{ji} = W_{ji}\,\frac{\sqrt{p_i^{\text{eq}}}}{\sqrt{p_j^{\text{eq}}}}
   = W_{ij}\,\frac{p_j^{\text{eq}}}{p_i^{\text{eq}}}\cdot\frac{\sqrt{p_i^{\text{eq}}}}{\sqrt{p_j^{\text{eq}}}}
   = W_{ij}\,\frac{\sqrt{p_j^{\text{eq}}}}{\sqrt{p_i^{\text{eq}}}} = \hat{W}_{ij}

using detailed balance in the second-to-last step. The diagonal loss and outflow terms
(``diag(O)`` and the rest of the diagonal of K) are left unchanged by a diagonal
similarity transform, :math:`D^{-1}\,\mathrm{diag}(\cdot)\,D = \mathrm{diag}(\cdot)`, so
the full matrix :math:`\hat{K} = D^{-1} K D` is symmetric whenever the thermal part
exactly satisfies detailed balance. Since K is similar to a symmetric matrix, it is
diagonalizable, with the same real eigenvalues as :math:`\hat{K}` (its eigenvectors are D
times those of :math:`\hat{K}`). Adding driven transitions D that break detailed balance,
or highly asymmetric loss terms, can move K away from this regime and toward a defective
(non-diagonalizable) matrix.

**What happens if K is not diagonalizable.** If K is defective (a repeated eigenvalue
without a matching number of independent eigenvectors, as near an exceptional point), S
is singular in the exact limit; approaching that limit, S is merely ill-conditioned. This
is what ``_warn_if_eig_basis_is_ill_conditioned`` checks for, by comparing
``torch.linalg.cond(eig_vecs)`` against a threshold (:math:`10^6` for float32,
:math:`10^{10}` for float64, :math:`10^{8}` otherwise). This matters because the error of
any quantity computed from the decomposition - in particular :math:`c = S^{-1} n(0)` and
therefore :math:`\exp(Kt)\, n(0)` - is amplified relative to machine precision roughly by
the condition number of S:

.. math::

   \frac{\lVert \text{computed} - \text{exact} \rVert}{\lVert \text{exact} \rVert}
   \;\sim\; \mathrm{cond}(S)\cdot \varepsilon_{\text{machine}}

so as K approaches a defective matrix, :math:`\mathrm{cond}(S) \to \infty` and the
eigen-based solution can lose all significant digits even though the underlying floating
point arithmetic itself is exact. Physically this also reflects a real property of the
kinetic model, not just a numerical artifact: near an exceptional point, tiny changes in
the rate constants can strongly change relaxation times and mode amplitudes, so the model
itself is structurally sensitive to the (experimentally uncertain) rate parameters. In
this regime, use ``EvolutionPopulationSolver.stationary_rate_solver_expm``, which computes
:math:`\exp(Kt)` directly (via matrix exponentiation, without eigen-decomposition) and
remains mathematically valid for defective matrices, at the cost of being slower for many
time points since a full :math:`N \times N` matrix exponential is evaluated per time
point instead of an N-vector of eigenvalues.

Usage Examples
--------------

Both classes referenced in this document can be imported from ``mars.population``: the
solvers live in ``mars.population.tr_utils``, and the populator lives in
``mars.population.populators.level_population``.

.. code-block:: python

   from mars.population import tr_utils
   from mars.population.populators import level_population

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
   populator = level_population.LevelBasedPopulator(
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

   populator = level_population.LevelBasedPopulator(
       context=context,
       solver=tr_utils.EvolutionPopulationSolver.exponential_solver,
       init_temperature=293.0,
   )

This increases memory overhead, because the matrix exponentials for all time intervals are
held in memory at once (shape ``[T-1, ..., N, N]``), but reduces total computation time
compared to the adaptive ODE integrator, since it replaces many small adaptive steps with
one batched matrix-exponential call. It is only a valid approximation while

.. math::

   \frac{1}{\lVert K \rVert}\left\lVert \frac{dK}{dt} \right\rVert
   \;\ll\; \left|\mathrm{Re}(\lambda_{\min})\right|

i.e. while the relaxation parameters change slowly compared to the relaxation process
itself; if K varies on a timescale comparable to or faster than relaxation, use
``odeint_solver`` instead, or refine the time grid.

Limitations
-----------

This approach: Treats only populations (diagonal density matrix elements)
For systems where coherences are important, use the density matrix approaches (RWA or propagator computation method).