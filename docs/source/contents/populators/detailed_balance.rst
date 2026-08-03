.. _detailed_balance:

Detailed Balance Enforcement
============================

Overview
--------

Spontaneous (thermal) relaxation transitions must satisfy the principle of
detailed balance at thermal equilibrium. This ensures that at temperature
:math:`T`, the system reaches a stationary Boltzmann distribution where the
rates of forward and backward transitions between any pair of energy levels
satisfy:

.. math::

   \frac{w_{j \to i}}{w_{i \to j}} = \exp\left(-\frac{E_i - E_j}{k_B T}\right)

In MaRs, the kinetic matrix element :math:`w_{ij}` is defined as the physical
transition rate **from state :math:`j` to state :math:`i`**, i.e.,

.. math::

   w_{ij} \equiv w_{j \to i}

Thus, the detailed balance condition can be equivalently written as:

.. math::

   \frac{w_{ij}}{w_{ji}} = \exp\left(-\frac{E_i - E_j}{k_B T}\right)

MaRs automatically enforces detailed balance for all thermal transition rates
by applying Boltzmann corrections.  Driven (induced) transitions are not
subject to this constraint and are added to the kinetic matrix or relaxation
superoperator without modification.

This document describes how MaRs modifies thermal transitions to satisfy
detailed balance in both the kinetic (population‑based) and density‑matrix
paradigms, and how the same corrections are applied to spectral densities used
in Redfield relaxation channels.


Detailed Balance in the Kinetic Approach
----------------------------------------

Input Convention
~~~~~~~~~~~~~~~~

The input matrix of thermal transition rates, denoted ``thermal_rates``, may
be *arbitrary*.  MaRs provides two modes of processing this input, controlled
by the flag ``thermal_balance_mode``:

1. **Symmetric mode** (``"symmetric"``, default for ``EvolutionMatrix``):
   The class **symmetrizes** the input internally by computing

   .. math::

      w'_{ij} = \frac{1}{2}(w_{ij} + w_{ji})

   This symmetric average is then used as the base rate for Boltzmann
   correction.  This mode is appropriate when the user provides raw or
   unstructured rates and wishes MaRs to enforce physical symmetry before
   thermal scaling.

2. **Complement mode** (``"complement"``):
   The input matrix is interpreted directly as physical rates
   :math:`w_{ij} = w_{j \to i}`.  Missing backward rates
   (where :math:`w_{ji} = 0` but :math:`w_{ij} > 0`) are inferred via
   detailed balance:

   .. math::

      w_{ij} = w_{ji} \cdot \exp\left(-\frac{E_i - E_j}{k_B T}\right)

3. **Skip mode** (``"skip"``):  No thermal correction is applied.

Boltzmann Correction (Symmetric Mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under symmetric mode, for each pair of energy levels :math:`i` and :math:`j`
with energy difference :math:`\Delta E_{ij} = E_j - E_i`, the corrected
transition rates are:

.. math::

   w_{ij}^* = \frac{2w'_{ij}}{1 + \exp(-\Delta E_{ij} / k_B T)}

.. math::

   w_{ji}^* = \frac{2w'_{ij}}{1 + \exp(\Delta E_{ij} / k_B T)}
            = w_{ij}^* \exp(-\Delta E_{ij} / k_B T)

where :math:`w'_{ij}` is the symmetric input rate (equal to :math:`w'_{ji}`).

**Verification**: The corrected probabilities satisfy:

.. math::

   \frac{w_{ij}^*}{w_{ji}^*} = \exp\left(-\frac{E_i - E_j}{k_B T}\right)
                             = \exp\left(\frac{\Delta E_{ij}}{k_B T}\right)

and their sum is conserved:

.. math::

   w_{ij}^* + w_{ji}^* = 2w'_{ij}

Kinetic Matrix Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full kinetic matrix :math:`K` is constructed as:

.. math::

   K = W^* + D - \operatorname{diag}(O)

where:

- :math:`W^*` is the matrix of Boltzmann‑corrected thermal transition rates
  (free probabilities).
- :math:`D` is the matrix of driven transitions (no Boltzmann correction).
- :math:`O` is the vector of outgoing loss rates (e.g., phosphorescence).

The diagonal elements of :math:`W^*` and :math:`D` are set to enforce
probability conservation:

.. math::

   W_{ii} = -\sum_{j \neq i} W_{ji}, \qquad
   D_{ii} = -\sum_{j \neq i} D_{ji}

Thus the total kinetic matrix has:

.. math::

   K_{ii} = -\sum_{j \neq i} (W_{ji} + D_{ji}) - O_i

This ensures that in the absence of losses (:math:`O = 0`), the column sums
are zero and total population is conserved.


Detailed Balance in the Density Matrix Approach
-----------------------------------------------

Liouville Space Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the density matrix formalism, the evolution equation in Liouville space is:

.. math::

   \frac{d\hat{\rho}}{dt} =
   \bigl(-i\hat{\mathcal{H}} + \hat{\mathcal{R}}_{\text{thermal}}
    + \hat{\mathcal{R}}_{\text{driv}}\bigr)\,\hat{\rho}

where:

- :math:`\hat{\mathcal{H}}` is the Hamiltonian superoperator:
  :math:`\hat{\mathcal{H}} = H \otimes I - I \otimes H`.
- :math:`\hat{\mathcal{R}}_{\text{thermal}}` is the spontaneous relaxation
  superoperator (subject to detailed balance).
- :math:`\hat{\mathcal{R}}_{\text{driv}}` is the driven relaxation
  superoperator (no detailed balance).

The density matrix :math:`\rho` (:math:`N \times N`) is vectorized into
:math:`\hat{\rho}` (:math:`N^2 \times 1`), and superoperators are
:math:`N^2 \times N^2` matrices.

Population Transfer Block
~~~~~~~~~~~~~~~~~~~~~~~~~

The relaxation superoperator couples elements of the density matrix. For
population transfers, only the diagonal block matters:

.. math::

   \hat{\mathcal{R}}[\,|i\rangle\langle i|,\,
                     |j\rangle\langle j|\,] \equiv \mathcal{R}_{iijj}

This element represents the rate of transition from population
:math:`\rho_{jj}` **to population** :math:`\rho_{ii}`.  In other words,
:math:`\mathcal{R}_{iijj}` is the rate :math:`j \to i`.

Detailed Balance Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

MaRs applies Boltzmann correction only to the population‑population coupling
block of the thermal relaxation superoperator.  Additionally, the **full
superoperator diagonal** is adjusted to preserve the decay rates and the
contribution of population loss to dephasing rates.  The algorithm is:

1. **Identify population indices** – For an :math:`N`-level system, the
   population elements of the vectorized density matrix are at positions
   :math:`\{0, N+1, 2(N+1), \dots, (N-1)(N+1)\}`, corresponding to
   :math:`\rho_{00}, \rho_{11}, \dots, \rho_{N-1,N-1}`.

2. **Extract the population submatrix** – Define the :math:`N \times N`
   matrix :math:`\mathbf{P}` by

   .. math::

      P_{ij} = \mathcal{R}_{iijj} \qquad (\text{rate from } j \text{ to } i),
      \quad i,j = 0,\dots,N-1.

3. **Store the original column sums** – For each level :math:`j`, compute
   the total outflow (including any irreversible losses) as the negative of
   the column sum:

   .. math::

      \text{colsum}_j = \sum_{i} P_{ij}.

   In a closed system :math:`\text{colsum}_j = 0`; with losses
   :math:`\text{colsum}_j = -\text{loss}_j`.  The values
   :math:`\text{colsum}_j` must be preserved after correction.

4. **Symmetrize and apply Boltzmann factor** – For every pair :math:`i \neq j`
   with energy difference :math:`\Delta E_{ij} = E_j - E_i`,
   compute the symmetric average :math:`s_{ij} = (P_{ij} + P_{ji})/2`. Then
   define the new rates that satisfy detailed balance:

   .. math::

      P'_{ij} = \frac{2 s_{ij}}{1 + \exp(-\Delta E_{ij}/k_B T)}, \qquad
      P'_{ji} = \frac{2 s_{ij}}{1 + \exp(\Delta E_{ij}/k_B T)}.

   These expressions preserve the sum :math:`P'_{ij} + P'_{ji} = 2 s_{ij}`
   and guarantee

   .. math::

      \frac{P'_{ij}}{P'_{ji}} =
      \exp\!\left(-\frac{E_i - E_j}{k_B T}\right).

5. **Restore the original column sums** – For each column :math:`j`,
   compute the sum of the new off‑diagonals:

   .. math::

      S'_j = \sum_{i \neq j} P'_{ij}.

   Then set the diagonal element so that the column sum remains unchanged:

   .. math::

      P'_{jj} = \text{colsum}_j - S'_j.

   This is equivalent to
   :math:`P'_{jj} = P_{jj} + \bigl(\sum_{i \neq j} P_{ij} - \sum_{i \neq j} P'_{ij}\bigr)`,
   ensuring that the total decay rate (including losses) from level :math:`j`
   is preserved.

6. **Correct the full superoperator diagonal** – The previous step adjusts
   only the population‑block diagonal, but the dephasing rates
   :math:`\Gamma_{ab} = \tfrac{1}{2}(\Gamma_a + \Gamma_b)` depend on the
   total outflow rates :math:`\Gamma_a = -\sum_i P_{ia}`.  After the
   population off‑diagonal elements are modified, the outflow rates change,
   and the corresponding dephasing rates must be updated.  MaRs computes a
   correction vector

   .. math::

      \Delta \Gamma_a = \sum_i (P_{ia} - P'_{ia})

   and then adjusts the diagonal entries of the full :math:`N^2 \times N^2`
   superoperator by

   .. math::

      \mathcal{R}_{(ab)(ab)} \to
      \mathcal{R}_{(ab)(ab)} - \tfrac{1}{2}(\Delta\Gamma_a + \Delta\Gamma_b).

   This preserves the original column sums of the population block and keeps
   the dephasing rates physically consistent with the new population transfer
   rates.

7. **Replace the population block** – Insert the corrected :math:`\mathbf{P}'`
   back into the full superoperator at the population indices. All
   coherence‑related off‑diagonal elements (coherence transfer,
   coherence‑population coupling) remain untouched.

**Result**: The corrected superoperator satisfies detailed balance for
population transfers:

.. math::

   \frac{\mathcal{R}_{iijj}^*}{\mathcal{R}_{jjii}^*} =
   \exp\!\left(-\frac{E_i - E_j}{k_B T}\right),

while the total outflow from each level (the column sum) and the dephasing
rates are identical to those of the input superoperator. This guarantees
that observable decay rates (e.g., spontaneous emission or phosphorescence)
are not artificially altered by the thermal correction.


Spectral Density Correction for Redfield Relaxation
---------------------------------------------------

When a :class:`~mars.population.relaxation.RedfieldRelaxationChannel` is used,
the spectral density :math:`J(\omega)` must also obey detailed balance to
produce correct upward and downward transition rates.  The
:meth:`~mars.population.thermal_corrections.ThermalBalanceCorrector.apply_matrix_transform`
method is applied directly to the spectral density matrix
:math:`J_{ij} = J(\omega_{ji})` (with :math:`\omega_{ji} = E_j - E_i` in
rad/s).  The correction behaves exactly as described for the kinetic rate
matrix, guaranteeing

.. math::

   J(-\omega) = J(\omega) \, e^{-\hbar\omega/k_B T}

when ``thermal_balance_mode`` is ``"symmetric"`` or ``"complement"``.
This ensures that the Redfield tensor generated from the corrected spectral
density automatically fulfills detailed balance at the given temperature.


Implementation in MaRs
-----------------------

The enforcement of detailed balance for spontaneous relaxation is centralized
in the class
:class:`~mars.population.thermal_corrections.ThermalBalanceCorrector`
(located in ``mars.population.thermal_corrections``).  It provides:

- :meth:`~mars.population.thermal_corrections.ThermalBalanceCorrector.apply_matrix_transform`
  – corrects a transition rate matrix or a spectral density matrix.
- :meth:`~mars.population.thermal_corrections.ThermalBalanceCorrector.apply_superoperator_transform`
  – corrects the population block of a Liouville‑space relaxation
  superoperator and updates the full diagonal.

This class is used internally by :class:`~mars.population.tr_utils.EvolutionMatrix`
(for kinetic matrix construction) and by
:class:`~mars.population.tr_utils.EvolutionSuper` (for density‑matrix
superoperator construction).  It is also called directly by the Redfield
channel when building the spectral density matrix.

The user selects the desired mode via the ``thermal_balance_mode`` argument
(either a string or a
:class:`~mars.population.thermal_corrections.ThermalBalanceMode` enum value:
``"skip"``, ``"symmetric"``, or ``"complement"``).