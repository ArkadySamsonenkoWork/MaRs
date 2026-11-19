.. _epr_spectrum_construction:

EPR Spectrum Construction
=========================

.. image:: /_static/mars_structure.png
   :width: 100%
   :alt: addition context
   :align: center


Resonance-based EPR spectrum construction in MaRs follows the steps below. For direct calculations, :class:`mars.spectra_manager.spectra_manager_direct.BaseDirectSpectra` evaluates the selected Hamiltonian at supplied magnetic field points and averages the computed signal over orientations, bypassing resonance-line searching, stick-spectrum assembly, and line broadening (see :ref:`direct-spectra`).

1. **Define spin particles**  
   The spin system consists of particles: electrons and nuclei - each specified by its spin quantum number and intrinsic magnetic parameters.

2. **Specify interactions**  
   Interactions (e.g., hyperfine, zero-field splitting, nuclear-nuclear couplings) are defined using the :class:`mars.spin_model.Interaction` class.
   (See :ref:`interaction_in_mars` for details).

3. **Assemble the spin system and sample**  
   Particles and interactions are combined into a :class:`mars.spin_model.SpinSystem`.
   Disordered (powder) samples are represented by :class:`mars.spin_model.SolidSample`.
   The resonance-based workflow is managed by the abstract class :class:`mars.spectra_manager.spectra_manager.BaseResSpectra`.

4. **Construct Hamiltonian matrices**  
   MaRs computes four matrices that define the spin Hamiltonian under an external magnetic field **B** = (B_X, B_Y, B_Z):

   .. math::
      \hat{H} = \hat{F} + B_X \hat{G}_X + B_Y \hat{G}_Y + B_Z \hat{G}_Z

   - :math:`\hat{F}`: Field-independent term (zero-field splitting, hyperfine, nuclear couplings, etc.)
   - :math:`\hat{G}_X, \hat{G}_Y, \hat{G}_Z`: Zeeman coupling operators, the derivatives of the Hamiltonian (in Hz) with respect to the magnetic field, :math:`\hat{G}_\alpha \equiv \partial \hat{H} / \partial B_\alpha = -\hat{\mu}_\alpha / h`, where :math:`\hat{\mu}_\alpha` is the magnetic dipole moment operator. Defined explicitly as:

     .. math::
        \hat{G}_\alpha = \frac{\mu_B}{h} \sum_{e} \sum_{\beta} g^{(e)}_{\alpha\beta} \hat{S}^{(e)}_\beta
                        - \frac{\mu_N}{h} \sum_{n} g_n \hat{I}^{(n)}_\alpha, \quad \alpha \in \{X, Y, Z\}

     where:
       - The first sum runs over all electrons (*e*), with :math:`g^{(e)}_{\alpha\beta}` the components of the electron g-tensor,
       - :math:`\hat{S}^{(e)}_\beta` is the β-component of the electron spin operator,
       - The second sum runs over all nuclei (*n*),
       - :math:`g_n` is the g-factor of nucleus *n* and :math:`\mu_N` is the nuclear magneton,
       - :math:`\hat{I}^{(n)}_\alpha` is the α-component of the nuclear spin operator.

5. **Determine resonance conditions**  
   MaRs offers three algorithms:

   5.1. **Resonance-field search** (:class:`mars.res_spectra_manager.res_line_solvers.field_algorithm.ResField`)  
        This method solves :math:`\hbar \omega = E_i(B_{ij}) - E_j(B_{ij})` for resonance fields :math:`B_{ij}` at fixed frequency.  
        This is a batched, GPU-accelerated implementation of the method used in EasySpin [doi:10.1016/j.jmr.2005.08.013]. In MaRs this method is default for solid-state simulations.

   5.2. **Secular approximation** (:meth:`mars.spin_model.BaseSample.get_hamiltonian_terms_secular`)
        The secular Hamiltonian is projected onto sectors of the total spin components,
        including both electron and nuclear spins:

        .. math::

           \hat{S}^{\mathrm{tot}}_\alpha
           = \sum_e \hat{S}^{(e)}_\alpha + \sum_n \hat{I}^{(n)}_\alpha,
           \qquad \alpha\in\{X,Y,Z\}.

        The projector :math:`\mathcal{P}_Q(A)=\sum_m P_m A P_m` retains matrix
        elements between states with equal eigenvalues of :math:`Q`, including
        entire degenerate total-projection sectors.

        1. The field-independent term :math:`\hat F` and longitudinal Zeeman
           operator :math:`\hat G_Z` are projected using :math:`\hat S_Z^{\mathrm{tot}}`.
           Therefore :math:`[\hat F_{\mathrm{sec}},\hat S_Z^{\mathrm{tot}}]=0` and
           :math:`[\hat G_{Z,\mathrm{sec}},\hat S_Z^{\mathrm{tot}}]=0`.

        2. The transverse Zeeman operators :math:`\hat G_X,\hat G_Y` are projected
           using :math:`\hat S_X^{\mathrm{tot}},\hat S_Y^{\mathrm{tot}}`, respectively.
           Their secular forms commute with those corresponding components.

        This does not generally imply :math:`[\hat F_{\mathrm{sec}},\hat G_{Z,\mathrm{sec}}]=0`.
        Secularization is used by default in
        :class:`mars.spectra_manager.DensityTimeSpectra` for time-resolved EPR.

   5.3. **Resonance-frequency search** (:class:`mars.spectra_manager.res_freq_algorithm.ResFreq`)  
        This method finds resonant frequencies in a given interval at fixed magnetic field. It requires only one matrix eigenvalues and eigenvectors computation

6. **Compute transition intensities**  
   Intensities are computed by :class:`mars.spectra_manager.intensity_base.BaseResIntensityCalculator`.

   Assuming the quantization axis aligns with the magnetic field direction (e.g., :math:`\mathbf{B} \parallel Z`), the intensity of a transition in stationart EPR spectroscopy between eigenstates :math:`|i\rangle` and :math:`|j\rangle` is:

   .. math::

      I_{ij} \propto \left| \langle i | \hat{G}_\perp | j \rangle \right|^2 (p_j - p_i)

   where :math:`\hat{G}_{\perp}` depends on the crystal or powder sample case, the population :math:`p_k` of state *k* for the equilibrium case is:

   .. math::

      p_k = \frac{e^{-E_k / k_B T}}{Z}, \quad Z = \sum_k e^{-E_k / k_B T} 

   In the crystal case, :math:`I_{ij} \propto \left| \langle i | \hat{G}_X | j \rangle \right|^2 \cdot (p_j - p_i)`.

   In disordered (powder) systems, the signal must be averaged over all molecular orientations.
   This is typically done by sampling three Euler angles (:math:'\alpha', '\beta', '\gamma') that define the rotation from the molecular (x, y, z) frame to the laboratory (X, Y, Z) frame.
   However, the final averaging over the third Euler angle (:math:'\gamma') can be performed analytically, reducing computational cost:
   
   .. math::
      I_{ij} \propto \frac{1}{2} \left( |\langle i | \hat{G}_X | j \rangle|^2 + |\langle i | \hat{G}_Y | j \rangle|^2 \right) \cdot (p_j - p_i)


   In **time-resolved** or **spin-polarized** simulations, populations :math:`p_i(t)` are not thermal and are managed via Context tool: see :class:`mars.population.contexts.Context`.

   For density matrix-based time dependant methods, the signal is computed directly as :math:`\mathrm{Tr}(\hat{G}_{\perp} \hat{\rho}(t))`,
   where :math:`\hat{G}_{\perp}` is the detected transverse component of the Zeeman operator (e.g., :math:`\hat{G}_X`, :math:`\hat{G}_Y`, or circular combinations depending on detection computation method)

7. **Account for line broadening**  
   The total Gaussian linewidth is computed as the square root of the sum of squares of all independent broadening contributions:

   .. math::
      \Gamma_{\text{Gauss}} = \sqrt{
          \Gamma_{\text{residual}}^2
          + \sum_{k} \Gamma_{\text{strain},k}^2
      }

   where:
   
   - :math:`\Gamma_{\text{residual}}` is the residual broadening due to unresolved interactions. It may be anisotropic and is specified as a full width at half maximum (FWHM) in *Hz*.
   - sum over strain contributions :math:`\sum_{k} \Gamma_{\text{strain},k}^2` arising from distributions in Hamiltonian parameters (e.g., g-tensor, zero-field splitting D/E, hyperfine tensors). Each :math:`\Gamma_{\text{strain},k}` is provided as FWHM in the natural units of the corresponding parameter:
     - dimensionless for the g-tensor,
     - in Hz for zero-field splitting parameters (D/E),

8. **Field-sweep Jacobian correction**  
   When converting from frequency-domain to field-swept spectra, intensities are scaled by :math:`1/|d\nu/dB|` to preserve spectral weight, consistent with standard EPR theory.

9. **Powder averaging with operation parallelism**  
   For powder samples, all matrix operations act on a batched tensor that combines orientation and sample dimensions. This enables full GPU parallelism across orientations without explicit loops.

10. **Interpolation to refined orientation grid**  
    Resonance positions, intensities, and linewidths are interpolated from a coarse to a fine spatial orientation grid to improve spectral resolution.

11. **Assemble stick spectrum**  
    A discrete spectrum is built from resonance lines with computed positions, intensities, and intrinsic widths.

12. **Apply lineshape convolution**  
    The final spectrum is obtained by convolving the stick spectrum with a Gaussian, Lorentzian, or Voigt profile (not pseudo-Voigt as implemented in some packages).

    Both Gaussian and Lorentzian broadening parameters are specified as full width at half maximum (FWHM) and measured according to the spectral domain:
    
    - In field-dependent (magnetic field sweep) simulations: widths are given in *tesla (T)*.
    - In frequency-dependent simulations: widths are given in *hertz (Hz)*.