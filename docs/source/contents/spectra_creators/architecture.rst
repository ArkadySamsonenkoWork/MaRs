Internal Architecture of Spectra Creators
=========================================

Spectra creators share :class:`mars.spectra_manager.spectra_manager.BaseSpectra`. The resonance-based branch, :class:`mars.spectra_manager.spectra_manager.BaseResSpectra`, orchestrates five modular stages:

1. **Resonance Solver**: Computes eigenvalues/eigenvectors of :math:`\hat{H}(B)` or :math:`\hat{H}(\omega)`.
2. **Intensity Calculator**: Evaluates transition matrix elements and weights by populations or density matrix.
3. **Linewidth Model**: Computes inhomogeneous broadening from strain tensors and mesh geometry.
4. **Spectral Integrator**: Performs orientation averaging (powder) or single-crystal projection.
5. **Post-Processor**: Applies final Gaussian and Lorentzian broadening

Each stage is replaceable via dependency injection (e.g., custom `intensity_calculator`), enabling advanced extensions without subclassing.

Direct Calculations
-------------------

The :class:`mars.spectra_manager.spectra_manager_direct.BaseDirectSpectra` branch diagonalizes the selected Hamiltonian at supplied field points, computes populations or density-matrix evolution through its intensity calculator, and averages the resulting signal over orientations. It bypasses resonance-line searching and spectral-line integration; broadening parameters are skipped. See :ref:`direct-spectra`.

Performance Notes
-----------------

- Use `inference_mode=True` (default) to disable gradient tracking for speed.
- For time-resolved simulations, caching (`recompute_spin_parameters=False`) avoids redundant diagonalization.

Extensibility
-------------

To implement a new creator:

- Subclass :class:`mars.spectra_manager.spectra_manager.BaseResSpectra` for resonance-line calculations, or :class:`mars.spectra_manager.spectra_manager_direct.BaseDirectSpectra` for direct fixed-field calculations
- Override `_init_spectra_processor` to select integration strategy
- Optionally override `_postcompute_batch_data` for custom dynamics