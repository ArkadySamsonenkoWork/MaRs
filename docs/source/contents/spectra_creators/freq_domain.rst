Frequency-Swept Stationary Spectra
==================================

While most EPR experiments sweep the magnetic field at fixed frequency, some setups (e.g., broadband spectrometers) sweep frequency at fixed field. The :class:`mars.spectra_manager.spectra_manager.StationaryFreqSpectra` class supports this mode.

It mirrors :class:`mars.spectra_manager.spectra_manager.StationarySpectra` but treats **frequency as the independent variable**. The resonance condition becomes:

.. math::

   \hbar \omega_{ij} = E_i(B_0) - E_j(B_0)

and the spectrum is computed as a function of :math:`\omega`.


Example
-------

.. code-block:: python

   g = spin_model.Interaction((2.0, 2.1, 2.2))
   zfs = spin_model.DEInteraction([10e9, 2e9])  # D=10 GHz
   sys = spin_model.SpinSystem(electrons=[1.0], g_tensors=[g], electron_electron=[(0,0,zfs)])
   sample = spin_model.SolidSample(sys)

   freq_creator = spectra_manager.StationaryFreqSpectra(field=1.0, sample=sample)  # B = 1 T
   freqs = torch.linspace(200e9, 400e9, 1000)  # 200–400 GHz
   spec = freq_creator(sample, freqs)