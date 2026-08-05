.. _relaxation_channels:

Relaxation Channels: Redfield and Lindblad
===========================================

MaRs provides built-in support for modelling spin relaxation via **Redfield** and **Lindblad** master equations.
Unlike the simple rate-matrix approach, these relaxation channels
define the system-bath coupling at the operator level and (in the Redfield case) incorporate a
frequency-dependent spectral density :math:`J(\omega)`.

A relaxation channel is specified as a list of operator tuples
:math:`\{(A_{\mathrm{static}}^{(k)}, A_{\mathrm{field}}^{(k)})\}`.
The total system-bath coupling Hamiltonian for channel :math:`k` is

.. math::
   H_{\mathrm{SB}}^{(k)} = \underbrace{A_{\mathrm{static}}^{(k)}}_{\text{dimensionless}}
                         + \underbrace{A_{\mathrm{field}}^{(k)}}_{\mathrm{T}^{-1}} \cdot B(t)

where :math:`B(t)` is the external magnetic field vector (in T).
If a term is not needed, the corresponding operator may be ``None``.

.. note::
   **Unit consistency** --
   The static coupling operator, :math:`A_{\mathrm{static}}` (``O_static``), must be dimensionless.
   The field-dependent coupling operator, :math:`A_{\mathrm{field}}` (``O_dependent``), must have units of :math:`\mathrm{T}^{-1}`.

   The Redfield tensor scales as :math:`|A|^2 J(\omega)`, so the product
   :math:`|A|^2 J(\omega)` must have units of :math:`\mathrm{rad\,s^{-1}}`.
   By default, :math:`|A|^2` is assumed to be dimensionless; therefore,
   :math:`J(\omega)` must be given in :math:`\mathrm{rad\,s^{-1}}`, producing rates
   :math:`W = |A|^2 J(\omega)` in :math:`\mathrm{rad\,s^{-1}}`.

   Alternatively, if :math:`A` is assigned units of :math:`\mathrm{rad\,s^{-1}}`,
   then :math:`J(\omega)` must have units of
   :math:`(\mathrm{rad\,s^{-1}})^{-1}` so that the product still has units of
   :math:`\mathrm{rad\,s^{-1}}`.

   For the Lindblad channel, the jump operators must already incorporate the rate scaling, i.e.
   :math:`L = \sqrt{\gamma}\,A`, where :math:`\gamma` is in :math:`\mathrm{rad\,s^{-1}}`.


Redfield Relaxation Channel
---------------------------

The Redfield channel (class :class:`~mars.population.relaxation_channels.redfield.RedfieldRelaxationChannel`)
solves the Bloch–Redfield master equation in the eigenbasis of the spin Hamiltonian.
It supports both secular and full (non‑secular) forms and can enforce detailed balance
via the :class:`~mars.population.thermal_balance.ThermalBalanceCorrector`.

Coupling Operators
^^^^^^^^^^^^^^^^^^

The argument ``operator_components`` is a list of tuples ``(static_op, field_op)``.
Each tuple defines one independent noise channel.
The operators must be square matrices of dimension :math:`N \times N`,
where :math:`N` is the Hilbert‑space dimension of the spin system.
They are stored in the original (working) basis and are transformed to the eigenbasis
automatically by the channel.

For example, to model relaxation driven by librations of the molecular frame
you can compute the derivative of the Hamiltonian with respect to a rotation
around a given axis (see `Tools for Constructing Coupling Operators`_ below).
The resulting ``O_static`` and ``O_dependent`` correspond exactly to the
entries of a tuple in ``operator_components``.

Spectral Density
^^^^^^^^^^^^^^^^

The Redfield channel requires a spectral density function

.. code:: python

   J(omega_rad_s: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor

- *omega_rad_s* : transition frequency in **rad/s**, can be positive or negative.
- *temperature* : temperature in Kelvin.
- Returns :math:`J(\omega)` with the same shape as *omega_rad_s*.
- The output must be in **rad/s** and should be non‑negative for positive frequencies.

A simple Ohmic spectral density with a Debye cutoff can be defined as:

.. code:: python

   def ohmic_debye(omega, temperature, eta=1.0, omega_c=1e12):
       """Ohmic spectral density with Debye cutoff (rad/s)."""
       return eta * omega / (1.0 + (omega / omega_c)**2)

Thermal Balance
^^^^^^^^^^^^^^^

The parameter ``thermal_balance_mode`` (default ``"skip"``) controls how detailed
balance :math:`J(-\omega) = J(\omega)e^{-\hbar\omega/k_BT}` is enforced.
Possible values are ``"skip"``, ``"symmetric"``, and ``"complement"``.
See :class:`~mars.population.thermal_balance.ThermalBalanceMode` for details.

Example: Librational Relaxation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   from mars.population import RedfieldRelaxationChannel

   # assume sample and axis already defined
   O_static, O_dependent = sample.get_librations_along_axis(axis)

   channel = RedfieldRelaxationChannel(
       operator_components=[(O_static, O_dependent)],
       spectral_density_func=ohmic_debye,
       thermal_balance_mode="symmetric"
   )

   # Pass to Context
   context = population.Context(
       sample=sample,
       basis="eigen",
       relaxation_coupling_channels=[channel],
       ...
   )

Lindblad Relaxation Channel
---------------------------

The Lindblad channel (class :class:`~mars.population.relaxation_channels.lindblad.LindbladRelaxationChannel`)
uses the standard Lindblad form

.. math::
   \frac{d\rho}{dt} = \sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\} \right)

No spectral density is needed; the jump operators :math:`L_k` must already include
the rate scaling (i.e. :math:`L_k = \sqrt{\gamma_k}A_k`).
As with the Redfield channel, the operators are built from tuples
``(static_op, field_op)`` that define the field‑independent and field‑dependent parts
of the coupling Hamiltonian. The channel internally forms the jump operator
:math:`L_k = A_{\mathrm{static}} + A_{\mathrm{field}}\cdot B`.

Thermal balance is applied to the population transfer rates extracted from the Lindblad
dissipator, supporting ``"skip"`` and ``"symmetric"`` modes.

.. note::
   **Distinction from kinetic rate matrices** –
   When you supply ``thermal_rates`` or ``driven_rates`` via the :class:`~mars.population.contexts.Context`
   interface, each non‑zero off‑diagonal entry :math:`W_{ij}` is internally promoted to an independent Lindblad
   jump operator :math:`|i\rangle\langle j|` with rate :math:`W_{ij}`.  The ``LindbladRelaxationChannel``,
   by contrast, lets you provide any matrix—not restricted to dyadic projector form—as the jump operator.

Example: Dephasing via Lindblad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   from mars.population import LindbladRelaxationChannel

   # Simple dephasing operator in eigenbasis (already in Hz^{1/2})
   Lz = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.complex128)) * (1e6 ** 0.5)

   channel = LindbladRelaxationChannel(
       operator_components=[(Lz, None)],  # no field dependence
       thermal_balance_mode="skip"
   )

   context = population.Context(
       sample=sample,
       basis="eigen",
       relaxation_coupling_channels=[channel],
       ...
   )

Tools for Constructing Coupling Operators
-----------------------------------------

MaRs provides a rich set of methods that directly produce the operator pairs
(or interaction tensors) needed for relaxation channels. They are grouped by use case.

Libration of the entire spin system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :meth:`mars.spin_model.MultiOrientedSample.get_librations_along_axis(axis)
  <mars.spin_model.MultiOrientedSample.get_librations_along_axis>`
  Returns a tuple ``(O_static, O_dependent)`` representing the derivative of the
  **full** spin Hamiltonian with respect to a small rotation around the given axis.
  ``O_static`` (Hz) comes from zero‑field terms (ZFS, hyperfine, dipolar),
  while ``O_dependent`` (in Hz / T) arises from the Zeeman interaction.
  This pair can be used directly as one entry of ``operator_components``.

Libration of a single interaction tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :meth:`Interaction.get_rotation_derivative_along_axis(axis)
  <mars.spin_model.Interaction.get_rotation_derivative_along_axis>`
  Computes the derivative of a single :class:`~mars.spin_model.Interaction` tensor
  (e.g., ZFS, hyperfine) with respect to rotation around *axis*.
  The result is a 3×3 tensor ``dQ/dθ`` that can then be oriented using the methods below.

Oriented interaction operators (all orientations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These methods belong to :class:`~mars.spin_model.MultiOrientedSample` and provide
the full spin operator for a given interaction tensor, rotated into the
laboratory frame and contracted with the appropriate spin operators.
They are essential for building field‑dependent coupling terms.

- :meth:`MultiOrientedSample.get_oriented_electron_electron_interaction(interaction, el_idx_1, el_idx_2)
  <mars.spin_model.MultiOrientedSample.get_oriented_electron_electron_interaction>`
  Returns the operator :math:`\hat{S}_1 \cdot \mathbf{Q} \cdot \hat{S}_2` for all orientations,
  where :math:`\mathbf{Q}` is the rotated interaction tensor.
- :meth:`MultiOrientedSample.get_oriented_electron_nuclei_interaction(interaction, el_idx, nuc_idx)
  <mars.spin_model.MultiOrientedSample.get_oriented_electron_nuclei_interaction>`
  Returns the hyperfine operator :math:`\hat{S} \cdot \mathbf{A} \cdot \hat{I}` for all orientations.
- :meth:`MultiOrientedSample.get_oriented_nuclei_nuclei_interaction(interaction, nuc_idx_1, nuc_idx_2)
  <mars.spin_model.MultiOrientedSample.get_oriented_nuclei_nuclei_interaction>`
  Returns the nuclear coupling operator :math:`\hat{I}_1 \cdot \mathbf{Q} \cdot \hat{I}_2`.
- :meth:`MultiOrientedSample.get_oriented_zeeman_interaction(interaction, el_idx)
  <mars.spin_model.MultiOrientedSample.get_oriented_zeeman_interaction>`
  Returns the Zeeman operator :math:`(\mu_B/h) \hat{S} \cdot \mathbf{g}` for all orientations.
  The result has an extra dimension for the three field components; contraction with
  :math:`\mathbf{B}` gives the full Zeeman term.

Using these methods to build a libration channel for a specific interaction:

1. Obtain the derivative tensor ``dQ_dtheta`` via
   :meth:`Interaction.get_rotation_derivative_along_axis(axis)
   <mars.spin_model.Interaction.get_rotation_derivative_along_axis>`.
2. Feed ``dQ_dtheta`` to the appropriate ``get_oriented_*`` method.
3. The returned operator can be used as ``O_static`` or ``O_dependent`` in a
   relaxation channel.

For example, to model libration of the ZFS interaction around the molecular *x*‑axis:

.. code:: python

   zfs_interaction = DEInteraction(100e6, 10e6) # the Interaction object
   dQ = zfs_interaction.get_rotation_derivative_along_axis(torch.tensor([1.,0.,0.]))
   O_static_zfs = sample.get_oriented_electron_electron_interaction(dQ, 0, 0)

   # This O_static_zfs can be the static part of a Redfield channel
   channel = RedfieldRelaxationChannel(
       operator_components=[(O_static_zfs, None)],
       spectral_density_func=my_spectral_density,
       thermal_balance_mode="symmetric"
   )

.. note::
   The orientation‑sensitive methods handle all mesh rotations automatically.
   The resulting operators already have the ``orientations`` dimension and are
   in the complex dtype.

Integration with Context
------------------------

Relaxation channels are passed to :class:`~mars.population.contexts.Context` via
the ``relaxation_coupling_channels`` argument (a list of
:class:`~mars.population.relaxation.BaseRelaxationChannel` instances).
They are evaluated together with any explicit rate matrices, dephasing,
or decay rates defined in the context, and the resulting superoperator is
transformed to the working basis of the simulation.

.. code:: python

   context = population.Context(
       sample=sample,
       basis="eigen",
       init_populations=[0.7, 0.2, 0.1],
       decay_rates=torch.tensor([1e2, 0.5e2, 0.8e2]),
       relaxation_coupling_channels=[redfield_channel, lindblad_channel],
       device=device, dtype=dtype
   )

All relaxation contributions (thermal rates, driven rates, decay, dephasing,
Redfield/Lindblad channels) are added coherently; see :ref:`complex_context`
for details on context algebra.