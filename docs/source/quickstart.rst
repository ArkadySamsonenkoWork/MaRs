Quickstart
==========

Simulate a basic EPR spectrum:

.. code-block:: python

   import torch
   import matplotlib.pyplot as plt
   from mars import spin_model, spectra_manager

   # Select device and precision
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   dtype = torch.float64

   # Define a simple electron spin system
   g_tensor = spin_model.Interaction((2.02, 2.04, 2.06), dtype=dtype, device=device)

   system = spin_model.SpinSystem(
       electrons=[0.5],
       g_tensors=[g_tensor],
       dtype=dtype,
       device=device
   )

   # Create a powder sample
   sample = spin_model.SolidSample(
       base_spin_system=system,
       gauss=0.001,
       lorentz=0.001,
       dtype=dtype,
       device=device
   )

   # Create spectrum calculator
   spectra = spectra_manager.StationarySpectra(
       freq=9.8e9,
       sample=sample,
       dtype=dtype,
       device=device
   )

   # Magnetic field range
   fields = torch.linspace(0.33, 0.36, 1000, device=device, dtype=dtype)

   # Compute spectrum
   intensity = spectra(sample, fields)

   # Plot result
   plt.plot(fields.cpu(), intensity.cpu())
   plt.xlabel("Magnetic field (T)")
   plt.ylabel("Intensity (a.u.)")
   plt.title("Simulated CW EPR Spectrum")
   plt.show()