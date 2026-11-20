"""MaRs RWA calculation for the triplet–doublet pair.

Set DIRECT to True to use the experimental fixed-field calculation.
"""

from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from mars import mesher, population, spectra_manager, spin_model
from mars.spectra_manager.res_line_solvers.res_field_algorithm import EighEigenSolver


# Calculation settings
DIRECT = False
dtype = torch.float64
device = torch.device("cpu")

# Experimental parameters
FREQUENCY_HZ = 9.75e9
B1_PEAK_UT = 0.05
FIELD_STEP_G = 0.1
FIELD_RADIUS_MT = 14.0
DELAY_WINDOW_US = 3.0
DELAY_POINTS = 1000
GAUSSIAN_FWHM_T = 0.60e-3

# Spin Hamiltonian
G_REFERENCE = 2.0023
D_MHZ = 300.0
EXCHANGE_MHZ = 35.0
MU_B_HZ_T = 13.99624555e9

# Spin preparation and relaxation
ZFS_POPULATIONS = (0.20, 0.20, 0.60)  # Tx, Ty, Tz populations.
T1_US = 3.0  # Time constant per allowed population transfer.
T2_NS = 10.0  # Total coherence decay time.
PAIR_LIFETIME_US = 1.0  # Effective lifetime of each eigenstate of the pair.

# Powder grid
POWDER_GRID_FREQUENCY = 20


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
    g_tensor_doublet = spin_model.Interaction(G_REFERENCE, dtype=dtype, device=device)
    g_tensor_triplet = spin_model.Interaction(G_REFERENCE, dtype=dtype, device=device)
    zfs_interaction = spin_model.DEInteraction([D_MHZ * 1e6, 0.0], dtype=dtype, device=device)
    exchange_interaction = spin_model.Interaction(EXCHANGE_MHZ * 1e6, dtype=dtype, device=device)

    base_spin_system = spin_model.SpinSystem(
        electrons=[0.5, 1.0],
        g_tensors=[g_tensor_doublet, g_tensor_triplet],
        electron_electron=[(0, 1, exchange_interaction), (1, 1, zfs_interaction)],
        dtype=dtype,
        device=device,
    )
    mesh = mesher.DelaunayMesh(
        initial_grid_frequency=POWDER_GRID_FREQUENCY, dtype=dtype, device=device,
    )
    cos2 = np.cos(mesh.initial_grid[:, 1].numpy())**2
    _, triangles = mesh.post_mesh
    areas = mesh.areas().numpy()
    weights = np.bincount(
        triangles.numpy().ravel(),
        weights=np.repeat(areas / 3, 3),
        minlength=len(cos2),
    )
    weights /= weights.sum()
    lorentz_width = None
    if not direct:
        lorentz_width = 1 / (np.pi * (T2_NS * 1e-9) * G_REFERENCE * MU_B_HZ_T)
    triplet_doublet = spin_model.SolidSample(
        base_spin_system=base_spin_system,
        mesh=mesh,
        gauss=None if direct else GAUSSIAN_FWHM_T,
        lorentz=lorentz_width,
        dtype=dtype,
        device=device,
    )

    # 2. Prepare the unpolarized doublet and polarized triplet, then set relaxation.
    doublet_system = spin_model.SpinSystem(
        electrons=[0.5], g_tensors=[g_tensor_doublet], dtype=dtype, device=device,
    )
    triplet_system = spin_model.SpinSystem(
        electrons=[1.0], g_tensors=[g_tensor_triplet],
        electron_electron=[(0, 0, zfs_interaction)], dtype=dtype, device=device,
    )
    triplet = spin_model.SolidSample(
        base_spin_system=triplet_system, mesh=mesh, dtype=dtype, device=device,
    )
    doublet_state = population.Context(
        basis=doublet_system.get_product_state_basis(),
        init_populations=[0.5, 0.5],
        dtype=dtype,
        device=device,
    )
    triplet_state = population.Context(
        sample=triplet, basis="xyz", init_populations=ZFS_POPULATIONS,
        dtype=dtype, device=device,
    )
    precursor = doublet_state @ triplet_state

    # Project onto the coupled high-field states at the centre of the sweep.
    zero_field, _, _, zeeman_z = triplet_doublet.get_hamiltonian_terms_secular()
    _, eigenvectors = EighEigenSolver()(zero_field + centre_field_t * zeeman_z)
    init_populations = precursor.get_transformed_init_populations(eigenvectors.unsqueeze(-3))

    decay_rates = 1 / (PAIR_LIFETIME_US * 1e-6)
    transfer_rates = np.full((6, 6), 1 / (T1_US * 1e-6))
    np.fill_diagonal(transfer_rates, 0.0)
    transfer_rates = torch.as_tensor(transfer_rates, dtype=dtype, device=device)
    # MaRs adds the coherence decay from transfer and loss. Supply the remainder.
    dephasing = 1 / (T2_NS * 1e-9) - transfer_rates.sum(dim=0) - decay_rates

    context = population.Context(
        sample=triplet_doublet,
        basis="eigen",
        init_populations=init_populations,
        driven_rates=transfer_rates,
        decay_rates=[decay_rates] * 6,
        dephasing=dephasing,
        dtype=dtype,
        device=device,
    )
    populator = population.RWADensityPopulator(
        context=context,
        b1_field=b1_peak_t,
        dtype=dtype,
        device=device,
    )

    # 3. Compute the time-resolved spectrum in the rotating-wave approximation.
    if direct:
        spectrum_creator = spectra_manager.BaseDirectSpectra(
            freq=FREQUENCY_HZ,
            sample=triplet_doublet,
            populator=populator,
            harmonic=0,
            hamiltonian_mode="secular",
            output_eigenvector=True,
            dtype=dtype,
            device=device,
        )
        block_size = min(64, max(8, 512 // len(cos2)))
        field_blocks = torch.split(fields, block_size)
        spectra = [spectrum_creator(triplet_doublet, block, time_points).numpy() for block in field_blocks]
        spectrum = np.concatenate(spectra, axis=1)

        # Use the benchmark's common field convolution after powder averaging.
        sigma_points = GAUSSIAN_FWHM_T / (np.sqrt(8 * np.log(2)) * (fields_t[1] - fields_t[0]))
        spectrum = gaussian_filter1d(spectrum, sigma_points, axis=1)
    else:
        spectrum_creator = spectra_manager.DensityTimeSpectra(
            freq=FREQUENCY_HZ,
            sample=triplet_doublet,
            populator=populator,
            harmonic=0,
            hamiltonian_mode="secular",
            output_eigenvector=True,
            computational_details=spectra_manager.ComputationalDetails(
                integration_chunk_size=32,
            ),
            dtype=dtype,
            device=device,
        )
        spectrum = spectrum_creator(triplet_doublet, fields, time_points).numpy()

    return dict(
        fields_T=fields_t, delays_s=delays_s, spectrum=spectrum,
        cos2=cos2, weights=weights, wall_s=perf_counter() - started,
        b1_peak_t=b1_peak_t, t2_ns=T2_NS,
    )


def calculate(direct=DIRECT):
    """Return the field–time map and its axes, with field in tesla and time in seconds."""
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _calculate(direct)
    finally:
        torch.set_num_threads(original_threads)
        torch.set_default_dtype(original_dtype)


if __name__ == "__main__":
    result = calculate()
    method = "direct" if DIRECT else "resfields"
    np.savez(f"mars_{method}_triplet_doublet.npz", **result)
    print(f"Calculated {result['spectrum'].shape} map in {result['wall_s']:.3f} s")
