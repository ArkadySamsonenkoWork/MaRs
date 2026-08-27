.. _magnetization_computation:

Magnetic Transition Dipole Moment in MaRs
=========================================

Physical Basis
--------------

The interaction between microwave radiation and a spin system is described by:

.. math::
   \hat{H}_{\text{int}}(t) = -\hat{\boldsymbol{\mu}} \cdot \mathbf{B}_1(t),

where the *magnetic dipole operator* is:

.. math::
   \hat{\boldsymbol{\mu}} = -\mu_B \sum_e \mathbf{g}^{(e)} \cdot \hat{\mathbf{S}}^{(e)} + \mu_N \sum_n g_n \hat{\mathbf{I}}^{(n)}.

For a transition :math:`|i\rangle \to |j\rangle`, the *magnetic transition dipole moment vector* is defined as [Nehrkorn *et al.*, Phys. Rev. Lett. **114**, 010801 (2015)]:

.. math::
   \boldsymbol{\mu}_{ij} = \langle j | \hat{\boldsymbol{\mu}} | i \rangle.

This complex vector fully characterizes the transition coupling to electromagnetic radiation. Its magnitude determines transition strength; its direction and phase encode polarization and rotational sense.

Standard Resonator Geometry
---------------------------

In a conventional EPR resonator, the oscillating microwave magnetic field
:math:`\mathbf{B}_1` can be oriented either perpendicular or parallel to the
static magnetic field :math:`\mathbf{B}_0`.

For the standard perpendicular mode
(:math:`\mathbf{B}_1 \perp \mathbf{B}_0`), the relevant transition
magnetization is obtained from the transverse components:

.. math::

   D_{\perp}
   =
   |\langle j | \hat{G}_X | i \rangle|^2

or for powder this expresssion is averaged for the all orientations among 3 euler angles.

For the parallel mode
(:math:`\mathbf{B}_1 \parallel \mathbf{B}_0`), the longitudinal component is
used:

.. math::

   D_{\parallel}
   =
   |\langle j | \hat{G}_Z | i \rangle|^2.

General Excitation Geometry (Beam EPR)
--------------------------------------

Following Nehrkorn *et al.* [PRL 114, 010801 (2015)], for arbitrary polarization and propagation direction :math:`\mathbf{n}_k`, the transition weight is:

- **Linear polarization** (direction :math:`\mathbf{n}_1`):

  .. math::
     D = |\mathbf{n}_1^\top \boldsymbol{\mu}_{ij}|^2.

- **Unpolarized radiation**:

  .. math::
     D = \tfrac{1}{2} \left( |\boldsymbol{\mu}_{ij}|^2 - |\mathbf{n}_k^\top \boldsymbol{\mu}_{ij}|^2 \right).

- **Circular polarization** (handedness :math:`\pm`):

  .. math::
     D^\pm = D^{\text{un}} \pm 2\, \mathbf{n}_k^\top \left( \mathrm{Im}\,\boldsymbol{\mu}_{ij} \times \mathrm{Re}\,\boldsymbol{\mu}_{ij} \right).

These expressions are used by :class:`mars.spectra_manager.intensity_wave.WaveIntensityCalculator`.
The population difference multiplies :math:`D` as a separate prefactor, preserving the two-factor structure as long as coherences are neglected.

Powder Averaging
----------------

For disordered samples, angular integration yields closed forms involving :math:`\xi_1 = \mathbf{n}_1^\top \mathbf{n}_0` and :math:`\xi_k = \mathbf{n}_k^\top \mathbf{n}_0` (see Eq. 3 in Nehrkorn *et al.*). For example, in Voigt geometry with unpolarized radiation:

.. math::
   D^{\text{powder}} = \tfrac{1}{4} \left( |\boldsymbol{\mu}_{ij}|^2 + |\mathbf{n}_0^\top \boldsymbol{\mu}_{ij}|^2 \right).


Example: Unpolarized radiation in a powder sample (Voigt geometry)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

For an incident electromagnetic wave, define the radiation geometry with
``WaveMagnetizationConfig`` and pass it to the spectra calculator.

In Voigt geometry, the wave propagation direction is perpendicular to the
static magnetic field B0, so ``theta = pi / 2``.

.. code-block:: python

   magnetization_config = spectra_manager.WaveMagnetizationConfig(
       polarization=spectra_manager.Polarization.UNPOLARIZED,
       theta=math.pi / 2,  # k perpendicular to B0: Voigt geometry
   )

   intensity_calculator = spectra_manager.WaveIntensityCalculator(
       spin_system_dim=sample.spin_system_dim,
       disordered=True, # powder sample or sample.mesh.disordered,
       magnetization_config=magnetization_config,
       temperature=300.0,
       device=device,
       dtype=dtype,
   )