.. _superoperator_creation:

Lindblad Superoperator Construction
====================================

MaRs uses the *Lindblad master equation* formalism to construct physically valid relaxation superoperators from user-defined relaxation rates.
This approach guarantees that the density matrix remains positive semi-definite and trace-preserving (if decay rates equal to zero) during time evolution.

.. image:: /_static/context/lindblad_overview.png
   :width: 100%
   :alt: Lindblad formalism overview
   :align: center


The Lindblad Master Equation
-----------------------------

The time evolution of the density matrix :math:`\hat{\rho}(t)` is governed by the Liouville-von Neumann equation with relaxation:

.. math::

   \frac{d\hat{\rho}}{dt} = -i[\hat{H}, \hat{\rho}] + \hat{\mathcal{R}}[\hat{\rho}]

where:

- :math:`\hat{H}` is the spin Hamiltonian
- :math:`\hat{\mathcal{R}}` is the relaxation superoperator

The most general Markovian relaxation that preserves positivity and trace is given by the **Lindblad form**:

.. math::

   \hat{\mathcal{R}}[\hat{\rho}] = \sum_k \left( \hat{L}_k \hat{\rho} \hat{L}_k^\dagger - \frac{1}{2} \{\hat{L}_k^\dagger \hat{L}_k, \hat{\rho}\} \right)

where:

- :math:`\hat{L}_k` are **Lindblad operators** (jump operators)
- :math:`\{A, B\} = AB + BA` is the anticommutator

Physical Interpretation
------------------------

Each Lindblad term has two components:

1. **Jump term** :math:`\hat{L}_k \hat{\rho} \hat{L}_k^\dagger`: Describes the system transitioning to a new state

2. **Decay term** :math:`-\frac{1}{2} \{\hat{L}_k^\dagger \hat{L}_k, \hat{\rho}\}`: Ensures trace preservation by accounting for population loss from the initial state

Mapping Relaxation Parameters to Lindblad Operators
----------------------------------------------------

MaRs maps the four relaxation parameter types to specific Lindblad operators:

Decay Rates → Loss Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For population loss from level :math:`|i\rangle` at rate :math:`o_i`:

.. math::

   \hat{L}_i = \sqrt{o_i} \, |i\rangle\langle i|

**Relaxation contribution:**

.. math::

   \hat{\mathcal{R}}_{\text{out}}[\hat{\rho}] = \sum_i o_i \left( |i\rangle\langle i| \hat{\rho} |i\rangle\langle i| - \frac{1}{2} \{|i\rangle\langle i|, \hat{\rho}\} \right)

This simplifies to:

.. math::

   \hat{\mathcal{R}}_{\text{out}}[\hat{\rho}] = -\frac{1}{2} \sum_i o_i \, \{|i\rangle\langle i|, \hat{\rho}\}

**Implementation:** See :func:`mars.population.transform.Liouvilleator.anticommutator_superop_diagonal` for the anticommutator construction.

Thermal Rates → Spontaneous Transition Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For spontaneous transitions :math:`|j\rangle \to |i\rangle` at rate :math:`w_{ij}`:

.. math::

   \hat{L}_{ij} = \sqrt{w_{ij}} \, |i\rangle\langle j|

**Relaxation contribution:**

.. math::

   \hat{\mathcal{R}}_{\text{thermal}}[\hat{\rho}] = \sum_{i \neq j} w_{ij} \left( |i\rangle\langle j| \hat{\rho} |j\rangle\langle i| - \frac{1}{2} \{|j\rangle\langle j|, \hat{\rho}\} \right)

This can be written as:

.. math::

   \hat{\mathcal{R}}_{\text{thermal}}[\hat{\rho}] = \sum_{i \neq j} w_{ij} \left( |i\rangle\langle i| \rho_{jj} - \frac{1}{2} \{|i\rangle\langle i| + |j\rangle\langle j|, \hat{\rho}\} \right)

This term:

- Increases population of :math:`|i\rangle` at rate :math:`w_{ij} \rho_{jj}`
- Decreases population of :math:`|j\rangle` at rate :math:`\sum_i w_{ij} \rho_{jj}`


**Implementation:** See :func:`mars.population.transform.Liouvilleator.lindblad_dissipator_from_rates` for the full construction.

Driven Rates → Stimulated Transition Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For driven transitions :math:`|j\rangle \to |i\rangle` at rate :math:`d_{ij}`:

.. math::

   \hat{L}_{ij} = \sqrt{d_{ij}} \, |i\rangle\langle j|

**Relaxation contribution:**

.. math::

   \hat{\mathcal{R}}_{\text{driv}}[\hat{\rho}] = \sum_{i \neq j} d_{ij} \left( |i\rangle\langle j| \hat{\rho} |j\rangle\langle i| - \frac{1}{2} \{|j\rangle\langle j|, \hat{\rho}\} \right)

This term is same form as thermal rates, but not modified by detailed balance constraints.

**Implementation:** See :func:`mars.population.transform.Liouvilleator.lindblad_dissipator_from_rates` for the full construction.

Dephasing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For pure dephasing of level :math:`|i\rangle` at rate :math:`\gamma_i`:

.. math::

   \hat{L}_i = \sqrt{\gamma_i} \, |i\rangle\langle i|

**Relaxation contribution:**

.. math::

   \hat{\mathcal{R}}_{\text{deph}}[\hat{\rho}] = \sum_i \gamma_i \left( |i\rangle\langle i| \hat{\rho} |i\rangle\langle i| - \frac{1}{2} \{|i\rangle\langle i|, \hat{\rho}\} \right)

For off-diagonal elements (:math:`i \neq j`):

.. math::

   \frac{d\rho_{ij}}{dt}\Big|_{\text{deph}} = -\frac{\gamma_i + \gamma_j}{2} \, \rho_{ij}

**Implementation:** See :func:`mars.population.transform.Liouvilleator.lindblad_dephasing_from_rates` for the construction.

Separation into Free and Driven Components
-------------------------------------------

MaRs internally separates the total relaxation superoperator:

.. math::

   \hat{\mathcal{R}} = \hat{\mathcal{R}}_{\text{thermal}} + \hat{\mathcal{R}}_{\text{driv}}

where:

**Free relaxation** :math:`\hat{\mathcal{R}}_{\text{thermal}}`:

- Includes: decay_rates, thermal_rates, dephasing
- Modified them to enforce detailed balance at temperature :math:`T`

**Driven relaxation** :math:`\hat{\mathcal{R}}_{\text{driv}}`:

- Includes: driven_rates or user-defined relaxation superoperator
- This term remains as specified by user


Liouville Space Representation
-------------------------------

For numerical computation, the superoperator is represented in **Liouville space** where:

- Density matrix :math:`\hat{\rho}` (dimension :math:`N \times N`) → vector :math:`|\rho\rangle\rangle` (dimension :math:`N^2`)
- Superoperator :math:`\hat{\mathcal{R}}` → matrix (dimension :math:`N^2 \times N^2`)

The transformation is:

.. math::

   \hat{\mathcal{L}} = -i(\hat{H} \otimes \mathbb{I} - \mathbb{I} \otimes \hat{H}^T) + \hat{\mathcal{R}}

where :math:`\mathbb{I}` is the identity operator.

**Key Functions:**

- :func:`mars.population.transform.Liouvilleator.vec` - Convert density matrix to vector
- :func:`mars.population.transform.Liouvilleator.unvec` - Convert vector back to matrix
- :func:`mars.population.transform.Liouvilleator.hamiltonian_superop` - Create Hamiltonian superoperator

Custom Relaxation Operators: Lindblad and Redfield
--------------------------------------------------

Beyond the predefined kinetic‑rate parameters, MaRs offers two ways to
define relaxation at the operator level:

**1. Direct construction of a superoperator**
   You can construct an arbitary superoperator and pass it to the context directly
   :math:`\hat{L}` and pass it directly to :class:`~mars.population.contexts.Context`
   via the ``relaxation_superop`` argument.

   For example, to build the supeoperator via the Lindblad Jump operator directly:

   .. code-block:: python

      from mars.population.transform import Liouvilleator
      dissipator = Liouvilleator.lindblad_dissipator_from_operator(L)

   This implements

   .. math::
      \mathcal{D}[\hat{\rho}] = \hat{L} \hat{\rho} \hat{L}^\dagger
                              - \tfrac{1}{2}\{\hat{L}^\dagger \hat{L}, \hat{\rho}\}.

   - The input ``L`` can be any complex‑valued tensor of shape ``[..., d, d]``.
   - The output superoperator has shape ``[..., d**2, d**2]`` in Liouville space.
   - For multiple independent jump channels :math:`\{\hat{L}_k\}`, sum the
     individual dissipators:
     ``total_R = sum(Liouvilleator.lindblad_dissipator_from_operator(L_k) for L_k in operators)``.

   The resulting superoperator is trace‑preserving and, when combined with
   Hamiltonian evolution, positivity‑preserving.


2. **Redfield (or Lindblad) relaxation channels**
   If the relaxation mechanism is more naturally described by a set of
   coupling operators :math:`A^{(k)}` and a bath spectral density
   :math:`J(\omega)`, you can use a :class:`~mars.population.relaxation_channels.redfield.RedfieldRelaxationChannel`.
   In this case MaRs automatically builds the Redfield operator in the
   eigenbasis, respecting the secular approximation and detailed balance.

   Alternatively, you can supply the same operator tuples to a
   :class:`~mars.population.relaxation_channels.lindblad.LindbladRelaxationChannel`
   if the jump operators already contain the amplitude factor
   (i.e. :math:`L_k = \sqrt{\gamma_k} A_k`).

   These channels are passed to :class:`~mars.population.contexts.Context`
   via the ``relaxation_coupling_channels`` list and are combined with any
   other relaxation terms automatically.

   For example, to model librational relaxation:

   .. code-block:: python

      from mars.population import RedfieldRelaxationChannel

      # Compute the field‑dependent coupling operator pair
      O_static, O_dependent = sample.get_librations_along_axis(axis)

      channel = RedfieldRelaxationChannel(
          operator_components=[(O_static, O_dependent)],
          spectral_density_func=my_spectral_density,
          thermal_balance_mode="symmetric"
      )

      context = population.Context(
          sample=sample,
          basis="eigen",
          relaxation_coupling_channels=[channel],
          ...
      )

   See :ref:`relaxation_channels` for a complete description and the available
   helper methods.


References
----------

1. Lindblad, G. (1976). "On the generators of quantum dynamical semigroups." *Communications in Mathematical Physics*, 48(2), 119-130. https://doi.org/10.1007/BF01608499

2. Breuer, H.-P., & Petruccione, F. (2002). *The Theory of Open Quantum Systems*. Oxford University Press.
