"""Teacups 2.3.0 RWA calculation for the stable spin-polarized triplet."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from functools import partial
import os
from time import perf_counter

import numpy as np
from teacups import creators, hamiltonians, input_handler
from teacups import signals_and_processing as signals
from teacups.classes import Exp, SimOpt, Sys
from teacups.convolution import voigt_convolution
from teacups.grid import sophe_grid


# Experimental parameters
FREQUENCY_HZ = 9.75e9
B1_PEAK_UT = 0.05
FIELD_STEP_G = 0.1
FIELD_RADIUS_MT = 12.5
DELAY_WINDOW_US = 8.0
DELAY_POINTS = 1000
GAUSSIAN_FWHM_T = 0.60e-3

# Spin Hamiltonian
G_REFERENCE = 2.0023
D_MHZ = 300.0
MU_B_HZ_T = 13.99624555e9

# Spin preparation and relaxation
ZFS_POPULATIONS = (0.20, 0.20, 0.60)  # Tx, Ty, Tz populations.
T1_US = 3.0  # Time constant per allowed population transfer.
T2_NS = 10.0  # Total coherence decay time.

# Orientation grid and calculation settings
POWDER_NODES = 40
WORKERS = 2
FIELDS_PER_BLOCK = 96
TIME_FACTOR = 2 * np.pi  # Time conversion for the Teacups preset.


def calculate():
    """Return the field–time map and its axes in tesla and seconds."""
    centre_field_t = FREQUENCY_HZ / (G_REFERENCE * MU_B_HZ_T)
    field_step_t = FIELD_STEP_G * 1e-4
    half_count = int(np.ceil(FIELD_RADIUS_MT * 1e-3 / field_step_t))
    fields_t = centre_field_t + np.arange(-half_count, half_count + 1) * field_step_t
    delays_s = np.linspace(0.0, DELAY_WINDOW_US * 1e-6, DELAY_POINTS)
    b1_peak_t = B1_PEAK_UT / 1e6
    started = perf_counter()

    # 1. Define the spin system and its preparation.
    spin_system = Sys()
    spin_system.spin_system = "trip"
    spin_system.g_tri = [G_REFERENCE] * 3
    spin_system.D_tri, spin_system.E_tri = D_MHZ, 0.0
    spin_system.T_relax_1 = (T1_US * 1e-6) * TIME_FACTOR
    spin_system.T_relax_2 = T2_NS * 1e-9 * TIME_FACTOR
    spin_system.width_gauss = 0.0
    spin_system.sigma_time = 0.0

    # 2. Set the experimental parameters.
    experiment = Exp()
    experiment.B_z = fields_t * 1e3  # mT
    experiment.freq_mw = FREQUENCY_HZ
    experiment.B_mw = 0.5 * b1_peak_t * 1e3  # Rotating component in mT.
    experiment.t_scale = [0.0, float(delays_s[-1] * TIME_FACTOR)]
    experiment.t_points = len(delays_s)

    # 3. Choose the Liouville-space solver and project the precursor on the SOPHE grid.
    options = SimOpt()
    options.space = "liouville"
    options.grid = "sophe"
    options.sym = "Dooh"
    options.grid_points = POWDER_NODES
    options.cpu_cores = 1

    theta, _, weights = sophe_grid(options.grid_points, options.sym)
    cos2 = np.cos(theta)**2
    init_populations = triplet_preparation(cos2)

    # 4. Calculate the powder spectrum with the native T1/T2 model.
    spectrum = simulate_powder(spin_system, experiment, options, init_populations)
    solid_angle = weights.sum()
    spectrum /= solid_angle
    weights /= solid_angle

    # Apply Gaussian broadening along the field axis.
    sigma_points = GAUSSIAN_FWHM_T / (np.sqrt(8 * np.log(2)) * (fields_t[1] - fields_t[0]))
    spectrum = voigt_convolution(0.0, sigma_points, spectrum)
    return dict(
        fields_T=fields_t, delays_s=delays_s, spectrum=spectrum,
        cos2=cos2, weights=weights, wall_s=perf_counter() - started,
        b1_peak_t=b1_peak_t, t2_ns=T2_NS,
    )


def triplet_preparation(cos2):
    """Project the axial ZFS populations onto the high-field states."""
    pxy, _, pz = ZFS_POPULATIONS
    p_middle = pxy + (pz - pxy) * cos2
    p_outer = (1 - p_middle) / 2
    return np.stack((p_outer, p_middle, p_outer), axis=-1)


def simulate_powder(spin_system, experiment, options, init_populations):
    """Apply the Teacups solver to each field block."""
    solve_block = partial(
        powder_field_block,
        system=spin_system,
        experiment=experiment,
        options=options,
        populations=init_populations,
    )
    n_fields = len(experiment.B_z)
    workers = min(WORKERS, max(1, n_fields // 2))
    block_count = max(workers, (n_fields + FIELDS_PER_BLOCK - 1) // FIELDS_PER_BLOCK)
    blocks = np.array_split(experiment.B_z, block_count)
    if workers == 1:
        spectra = list(map(solve_block, blocks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            spectra = list(executor.map(solve_block, blocks))
    return np.concatenate(spectra, axis=1).astype(np.float64)


def powder_field_block(fields_mt, *, system, experiment, options, populations):
    """Evolve the density matrix and average the signal on the native grid."""
    with open(os.devnull, "w") as quiet, redirect_stdout(quiet):
        sys, exp, opt, cal = input_handler.input_object_handler(system, experiment, options)
        exp.B_z = fields_mt.copy()
        input_handler.hyperfine_converter(sys)
        input_handler.scale_inputs(sys, exp, opt)
        input_handler.predefinitions(sys, exp, cal)
        input_handler.initialize_spin_system(sys)
        input_handler.create_grid(opt, cal)
        creators.set_up_spinoperator(sys, cal)
        creators.set_up_observable(sys, opt, cal)
        creators.set_up_tensors(sys, cal)

        cal.ham_sys = hamiltonians.set_up_triplet_hamiltonian(exp, opt, cal)
        hamiltonian = cal.ham_sys + hamiltonians.set_up_mw_hamiltonian(sys, exp, opt, cal)
        vectors = np.linalg.eigh(cal.ham_sys)[1]
        # Embed the matched secular populations in the solver's basis.
        adjoint = vectors.conj().swapaxes(-1, -2)
        rho = (vectors * populations[None, :, None, :]) @ adjoint
        cal.rho = rho.reshape(*rho.shape[:2], -1).astype(np.complex64)

        cal.ham = hamiltonian
        cal.eigvec = vectors
        hamiltonians.set_up_commutator_superoperator(sys, opt, cal)
        del cal.ham_sys, cal.ham, cal.eigvec, hamiltonian, vectors

        signals.propagation(sys, opt, cal)
        del cal.ham_superop
        # Use the native single-process kernel inside each field worker.
        cal.signal = np.empty((len(cal.t), len(exp.B_z), opt.grid_points), dtype=np.complex64)
        signals.time_evolution_liouville(cal)
        signals.powder_average(opt, cal)
    return cal.spec_sim.real


if __name__ == "__main__":
    result = calculate()
    np.savez("teacups_triplet_stable_fast_t2.npz", **result)
    print(f"Calculated {result['spectrum'].shape} map in {result['wall_s']:.3f} s")
