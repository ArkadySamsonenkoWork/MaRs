"""MaRs RWA calculation for the spin-correlated radical pair.

Set DIRECT to True to use the experimental fixed-field calculation.
"""

from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from mars import constants, mesher, population, spectra_manager, spin_model


# Calculation settings
DIRECT = False
dtype = torch.float64
device = torch.device("cpu")

# Experimental parameters
FREQUENCY_HZ = 9.75e9
B1_PEAK_UT = 0.05
FIELD_STEP_G = 0.1
FIELD_RADIUS_MT = 4.0
DELAY_WINDOW_US = 8.0
DELAY_POINTS = 1000
GAUSSIAN_FWHM_T = 0.60e-3

# Spin Hamiltonian
G_REFERENCE = 2.0023
G_FACTORS = (2.0048, 1.9998)
EXCHANGE_MHZ = 30.0
MU_B_HZ_T = 13.99624555e9

# Spin preparation and relaxation
T1_US = 3.0  # Time constant per allowed population transfer.
T2_NS = 10.0  # Total coherence decay time.
INNER_T1_US = 1.0
PAIR_LIFETIME_US = 3.0  # Effective lifetime of each eigenstate of the pair.


def _calculate(direct):
    # Set the recording field and delay axes.
    centre_field_t = FREQUENCY_HZ / (G_REFERENCE * MU_B_HZ_T)
    field_step_t = FIELD_STEP_G * 1e-4
    half_count = int(np.ceil(FIELD_RADIUS_MT * 1e-3 / field_step_t))
    fields_t = centre_field_t + np.arange(-half_count, half_count + 1) * field_step_t
    delays_s = np.linspace(0.0, DELAY_WINDOW_US * 1e-6, DELAY_POINTS)
    fields = torch.as_tensor(fields_t, dtype=dtype, device=device)
    time_points = torch.as_tensor(delays_s, dtype=dtype, device=device)
    b1_peak_t = B1_PEAK_UT / 1e6
    started = perf_counter()

    # 1. Build the spin system and the sample.
    # Match the benchmark's common transverse Zeeman coefficient.
    g_transverse = G_REFERENCE * MU_B_HZ_T / (constants.BOHR / constants.PLANCK)
    g_tensor_1 = spin_model.Interaction(
        [g_transverse, g_transverse, G_FACTORS[0]], dtype=dtype, device=device,
    )
    g_tensor_2 = spin_model.Interaction(
        [g_transverse, g_transverse, G_FACTORS[1]], dtype=dtype, device=device,
    )
    exchange_interaction = spin_model.Interaction(EXCHANGE_MHZ * 1e6, dtype=dtype, device=device)

    base_spin_system = spin_model.SpinSystem(
        electrons=[0.5, 0.5],
        g_tensors=[g_tensor_1, g_tensor_2],
        electron_electron=[(0, 1, exchange_interaction)],
        dtype=dtype,
        device=device,
    )
    mesh = mesher.CrystalMesh(euler_angles=[[0.0, 0.0, 0.0]], dtype=dtype, device=device)
    cos2 = np.array([1.0])
    weights = np.array([1.0])
    lorentz_width = None
    if not direct:
        lorentz_width = 1 / (np.pi * (T2_NS * 1e-9) * G_REFERENCE * MU_B_HZ_T)
    radical_pair = spin_model.SolidSample(
        base_spin_system=base_spin_system,
        mesh=mesh,
        gauss=None if direct else GAUSSIAN_FWHM_T,
        lorentz=lorentz_width,
        dtype=dtype,
        device=device,
    )

    # 2. Set the singlet preparation and the eigenbasis relaxation.
    context_singlet = population.Context(
        sample=radical_pair,
        basis="multiplet",
        init_populations=[1.0, 0.0, 0.0, 0.0],
        dtype=dtype,
        device=device,
    )
    decay_rates = 1 / (PAIR_LIFETIME_US * 1e-6)
    transfer_rates = np.full((4, 4), 1 / (T1_US * 1e-6))
    np.fill_diagonal(transfer_rates, 0.0)
    # The two inner eigenstates mix S and T0; T+ and T- do not exchange directly.
    transfer_rates[1, 2] = transfer_rates[2, 1] = 1 / (INNER_T1_US * 1e-6)
    transfer_rates[0, 3] = transfer_rates[3, 0] = 0.0
    transfer_rates = torch.as_tensor(transfer_rates, dtype=dtype, device=device)
    # MaRs adds the coherence decay from transfer and loss. Supply the remainder.
    dephasing = 1 / (T2_NS * 1e-9) - transfer_rates.sum(dim=0) - decay_rates

    context_relaxation = population.Context(
        sample=radical_pair,
        basis="eigen",
        driven_rates=transfer_rates,
        decay_rates=[decay_rates] * 4,
        dephasing=dephasing,
        dtype=dtype,
        device=device,
    )
    context = context_singlet + context_relaxation
    populator = population.RWADensityPopulator(
        context=context,
        b1_field=b1_peak_t,
        dtype=dtype,
        device=device,
    )

    # 3. Compute the time-resolved spectrum in the rotating-wave approximation.
    # The direct static Hamiltonian retains the S/T0 mixing in both modes.
    if direct:
        spectrum_creator = spectra_manager.BaseDirectSpectra(
            freq=FREQUENCY_HZ,
            sample=radical_pair,
            populator=populator,
            harmonic=0,
            hamiltonian_mode="direct",
            output_eigenvector=True,
            dtype=dtype,
            device=device,
        )
        block_size = 64
        field_blocks = torch.split(fields, block_size)
        spectra = [spectrum_creator(radical_pair, block, time_points).numpy() for block in field_blocks]
        spectrum = np.concatenate(spectra, axis=1)

        # Use the benchmark's common field convolution after powder averaging.
        sigma_points = GAUSSIAN_FWHM_T / (np.sqrt(8 * np.log(2)) * (fields_t[1] - fields_t[0]))
        spectrum = gaussian_filter1d(spectrum, sigma_points, axis=1)
    else:
        spectrum_creator = spectra_manager.DensityTimeSpectra(
            freq=FREQUENCY_HZ,
            sample=radical_pair,
            populator=populator,
            harmonic=0,
            hamiltonian_mode="direct",
            output_eigenvector=True,
            dtype=dtype,
            device=device,
        )
        spectrum = spectrum_creator(radical_pair, fields, time_points).numpy()

    return dict(
        fields_T=fields_t, delays_s=delays_s, spectrum=spectrum,
        cos2=cos2, weights=weights, wall_s=perf_counter() - started,
        b1_peak_t=b1_peak_t, t2_ns=T2_NS,
    )


def calculate(direct=DIRECT):
    """Return the field–time map and its axes, with field in tesla and time in seconds."""
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _calculate(direct)
    finally:
        torch.set_num_threads(original_threads)


if __name__ == "__main__":
    result = calculate()
    method = "direct" if DIRECT else "resfields"
    np.savez(f"mars_{method}_radical_pair.npz", **result)
    print(f"Calculated {result['spectrum'].shape} map in {result['wall_s']:.3f} s")
