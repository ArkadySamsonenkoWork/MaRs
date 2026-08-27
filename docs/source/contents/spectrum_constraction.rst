.. _epr_spectrum_construction:

EPR Spectrum Construction
=========================

.. image:: /_static/mars_structure.png
   :width: 100%
   :alt: addition context
   :align: center


The construction of an EPR spectrum in MaRs follows a sequence of computational steps:

1. **Define spin particles**  
   The spin system consists of particles: electrons and nuclei - each specified by its spin quantum number and intrinsic magnetic parameters.

2. **Specify interactions**  
   Interactions (e.g., hyperfine, zero-field splitting, nuclear-nuclear couplings) are defined using the :class:`mars.spin_model.Interaction` class.
   (See :ref:`interaction_in_mars` for details).

3. **Assemble the spin system and sample**  
   Particles and interactions are combined into a :class:`mars.spin_model.SpinSystem`.
   Disordered (powder) samples are represented by :class:`mars.spin_model.MultiOrientedSample`.
   The next workflow is managed by the abstract class :class:`mars.spectra_manager.spectra_manager.BaseResSpectra`.

4. **Construct Hamiltonian matrices**  
   MaRs computes four matrices that define the spin Hamiltonian under an external magnetic field **B** = (Bx, By, Bz):

   .. math::
      \hat{H} = \hat{F} + B_x \hat{G}_X + B_y \hat{G}_Y + B_z \hat{G}_Z

   - **F**: Field-independent term (zero-field splitting, hyperfine, nuclear couplings, etc.)
   - **Gx, Gy, Gz**: Zeeman coupling operators, defined explicitly as:

     .. math::
        \hat{G}_\alpha = \mu_B \sum_{e} \sum_{\beta} g^{(e)}_{\alpha\beta} \hat{S}^{(e)}_\beta
                        + \sum_{n} \gamma_n \hat{I}^{(n)}_\alpha, \quad \alpha \in \{x, y, z\}

     where:
       - The first sum runs over all electrons (*e*), with :math:`g^{(e)}_{\alpha\beta}` the components of the electron g-tensor,
       - :math:`\hat{S}^{(e)}_\beta` is the β-component of the electron spin operator,
       - The second sum runs over all nuclei (*n*),
       - :math:`\gamma_n = g_n \mu_N / \hbar` is the gyromagnetic ratio of nucleus *n*, expressed via its g-factor :math:`g_n` and the nuclear magneton :math:`\mu_N`,
       - :math:`\hat{I}^{(n)}_\alpha` is the α-component of the nuclear spin operator.

5. **Determine resonance conditions**  
   MaRs offers three algorithms:

   5.1. **Resonance-field search** (:class:`mars.res_spectra_manager.res_line_solvers.field_algorithm.ResField`)  
        This method solves :math:`\hbar \omega = E_i(B_{ij}) - E_j(B_{ij})` for resonance fields :math:`B_{ij}` at fixed frequency.  
        This is a batched, GPU-accelerated implementation of the method used in EasySpin [doi:10.1016/j.jmr.2005.08.013]. In MaRs this method is default for solid-state simulations.

   5.2. **Secular approximation** (:class:`mars.secular_approximation_algorithm.ResSecular`)
        This method applies a two-step approximation to the spin Hamiltonian:
        
        1. Zeeman term projection: Modifies :math:`G_x, G_y, G_z` to zero matrix elements where the corresponding total spin projection (:math:`S_x, S_y, S_z`) has magnitude below threshold. Enforces :math:`[G_\alpha, S_\alpha] \approx 0`.
        
        2. Zero-field term projection: Modifies :math:`F` to zero elements connecting states with different Zeeman energies (:math:`|(G_z)_{ii} - (G_z)_{jj}| > \varepsilon`), enforcing :math:`[F, G_z] \approx 0`.
        
        This method is udsed by default in :class:`mars.spectra_manager.spectra_manager.DensityTimeSpectra` for time-resolved EPR.
   
   5.3. **Resonance-frequency search** (:class:`mars.spectra_manager.res_freq_algorithm.ResFreq`)  
        This method finds resonant frequencies in a given interval at fixed magnetic field. It requires only one matrix eigenvalues and eigenvectors computation

6. **Compute transition intensities**  
   Intensities are computed by :class:`mars.spectra_manager.spectra_manager.BaseResIntensityCalculator`.

   Assuming the quantization axis aligns with the magnetic field direction (e.g., :math:`\mathbf{B} \parallel z`), the intensity of a transition in stationart EPR spectroscopy between eigenstates :math:`|i\rangle` and :math:`|j\rangle` is:

   .. math::

      I_{ij} \propto \left| \langle i | \hat{G}_\perp | j \rangle \right|^2 (p_j - p_i)

   where :math:`\hat{G}_{\perp}` depends on the crystal or powder sample case, the population :math:`p_k` of state *k* for the equilibrium case is:

   .. math::

      p_k = \frac{e^{-E_k / k_B T}}{Z}, \quad Z = \sum_k e^{-E_k / k_B T} 

   In the crystal case, :math:`I_{ij} \propto \left| \langle i | \hat{G}_x | j \rangle \right|^2 \cdot (p_j - p_i)`.

   In disordered (powder) systems, the signal must be averaged over all molecular orientations.
   This is typically done by sampling three Euler angles (:math:'\alpha', '\beta', '\gamma') that define the rotation from the molecular frame to the lab frame.
   However, the final averaging over the third Euler angle (:math:'\gamma') can be performed analytically, reducing computational cost:
   
   .. math::
      I_{ij} \propto \frac{1}{2} \left( |\langle i | \hat{G}_x | j \rangle|^2 + |\langle i | \hat{G}_y | j \rangle|^2 \right) \cdot (p_j - p_i)


   In **time-resolved** or **spin-polarized** simulations, populations :math:`p_i(t)` are not thermal and are managed via Context tool: see :class:`mars.population.contexts.Context`.

   For density matrix-based time dependant methods, the signal is computed directly as :math:`\mathrm{Tr}(\hat{G}_{\perp} \hat{\rho}(t))`,
   where :math:`\hat{G}_{\perp}` is the detected transverse spin component (e.g., :math:`\hat{G}_x`, :math:`\hat{G}_y`, or circular combinations depending on detection computation method)

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
   When converting from frequency-domain to field-swept spectra, intensities are scaled by :math:`|d\nu/dB|` to preserve spectral weight, consistent with standard EPR theory.

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