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

In conventional EPR (:math:`\mathbf{B}_1 \perp \mathbf{B}_0`, linear polarization), the relevant quantity is:

.. math::
   D = |\langle j | \hat{G}_x | i \rangle|^2 + |\langle j | \hat{G}_y | i \rangle|^2,

which equals :math:`|\mathbf{n}_1^\top \boldsymbol{\mu}_{ij}|^2` for :math:`\mathbf{n}_1 \perp \mathbf{n}_0`.

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

These expressions are used by :class:`mars.spectra_manager.wave_calculator.WaveIntensityCalculator`.
The population difference multiplies :math:`D` as a separate prefactor, preserving the two-factor structure as long as coherences are neglected.

Powder Averaging
----------------

For disordered samples, angular integration yields closed forms involving :math:`\xi_1 = \mathbf{n}_1^\top \mathbf{n}_0` and :math:`\xi_k = \mathbf{n}_k^\top \mathbf{n}_0` (see Eq. 3 in Nehrkorn *et al.*). For example, in Voigt geometry with unpolarized radiation:

.. math::
   D^{\text{powder}} = \tfrac{1}{4} \left( |\boldsymbol{\mu}_{ij}|^2 + |\mathbf{n}_0^\top \boldsymbol{\mu}_{ij}|^2 \right).

Usage Examples

Example: Unpolarized radiation in a powder sample (Voigt geometry)
--------------------------------------------------------------------

.. code-block:: python

   calculator = specta_manager.WaveIntensityCalculator(
       spin_system_dim=sample.spin_system_dim,
       disordered=True,                  # powder sample or sample.mesh.disordered,
       polarization='un',                # unpolarized
       theta=math.pi / 2,                # propagation perpendicular to B0
       temperature=300.0,                # room temperature
       device=device,
       dtype=dtype
   )