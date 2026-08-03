Density-Matrix Time-Resolved Spectra
====================================

The time evolution of the spin density matrix :math:`\rho(t)` is governed by the **Liouville–von Neumann equation** with dissipation:

.. math::

   \frac{d\rho}{dt} = -\frac{i}{\hbar} [\hat{H}, \rho] + \mathcal{R}[\rho]

where :math:`\hat{H}` is the system Hamiltonian and :math:`\mathcal{R}[\rho]` is the relaxation superoperator.

The relaxation superoperator :math:`\mathcal{R}[\rho]` accounts for four physically distinct processes that govern population and coherence dynamics in time-resolved EPR (see more details in :ref:`relaxation_parameters`):

- **Free probabilities**: thermalizing transitions between eigenstates obeying detailed balance,
- **Driven probabilities**: coherent or incoherent microwave-induced transitions,
- **Out probabilities**: irreversible population loss (e.g., radiative or non-radiative decay),
- **Dephasing rates**: pure dephasing between energy levels.

These processes are encoded in the **Lindblad master equation**, which provides the explicit form of :math:`\mathcal{R}[\rho]`:

.. math::

   \mathcal{R}[\rho] = \sum_k \left( 
      \hat{L}_k \rho \hat{L}_k^\dagger 
      - \frac{1}{2} \{ \hat{L}_k^\dagger \hat{L}_k, \rho \}
   \right)

In addition to the four standard contributions, MaRs supports user-defined superoperators, allowing custom relaxation, or phenomenological models to be incorporated directly into :math:`\mathcal{R}[\rho]`.

Two computational strategies are available for time propagation:

- **Rotating Wave Approximation (RWA)**  
  The Hamiltonian is transformed into a frame rotating at the excitation frequency, and fast-oscillating terms are discarded.
  This yields a time-independent effective Liouvillian, enabling efficient simulation. (see details in :ref:`rotating-wave-approximation`)

- **Full Propagator Construction**  
  The complete Liouvillian is used to construct the exact evolution superoperator (see details in :ref:`propagator-computations`):

  .. math::

     \mathcal{U}(\Delta t) = \exp\!\big( \mathcal{L} \, \Delta t \big), \quad \text{where} \quad \mathcal{L}[\rho] = -\frac{i}{\hbar}[\hat{H}, \rho] + \mathcal{R}[\rho]

While RWA reduces computational cost, it suffers from limitations:

- The electron Zeeman interaction must be isotropic or close to isotropic: :math:`\hat{G}_x = g \mu_B \hat{S}_x`, etc.
- The static part of the Hamiltonian :math:`\hat{F}` (all terms except Zeeman) must commute with :math:`\hat{G}_z`: :math:`[\hat{F}, \hat{G}_z] = 0`.
- The relaxation superoperator only couples matrix elements :math:`\rho_{ij}` and :math:`\rho_{kl}` when :math:`i - j = k - l`.

All relaxation terms and jump operators alike are first defined in the user-specified basis (e.g., ``"zfs"``, ``"eigen"``) and then transformed into the Hamiltonian eigenbasis.
