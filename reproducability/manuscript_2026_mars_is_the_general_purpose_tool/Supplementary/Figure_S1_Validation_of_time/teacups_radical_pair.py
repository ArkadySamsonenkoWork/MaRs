"""Teacups 2.3.0 RWA calculation for the spin-correlated radical pair."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from functools import partial
import os
from time import perf_counter

import numpy as np
from teacups import creators, density_matrices, hamiltonians, input_handler, relaxation
from teacups import signals_and_processing as signals
from teacups.classes import Exp, SimOpt, Sys
from teacups.convolution import voigt_convolution
from teacups.grid import sophe_grid


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

# Orientation grid and calculation settings
WORKERS = 2
FIELDS_PER_BLOCK = 256
TIME_FACTOR = 1.0  # Time conversion for the Teacups preset.


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
    spin_system.spin_system = "rp"
    spin_system.precursor = "singlet"
    spin_system.g1, spin_system.g2 = ([g] * 3 for g in G_FACTORS)
    spin_system.J_ex = EXCHANGE_MHZ
    spin_system.D = spin_system.E = 0.0
    spin_system.width_gauss = 0.0
    spin_system.sigma_time = 0.0

    # 2. Set the experimental parameters.
    experiment = Exp()
    experiment.B_z = fields_t * 1e3  # mT
    experiment.freq_mw = FREQUENCY_HZ
    experiment.B_mw = 0.5 * b1_peak_t * 1e3  # Rotating component in mT.
    # The radical-pair microwave operator in Teacups includes sqrt(2).
    experiment.B_mw /= np.sqrt(2)
    experiment.t_scale = [0.0, float(delays_s[-1] * TIME_FACTOR)]
    experiment.t_points = len(delays_s)

    # 3. Choose the Liouville-space solver and set up relaxation on the SOPHE grid.
    options = SimOpt()
    options.space = "liouville"
    options.grid = "sophe"
    options.sym = "O3"
    options.grid_points = 2  # The isotropic SOPHE grid has one orientation.
    options.cpu_cores = 1

    theta, _, weights = sophe_grid(options.grid_points, options.sym)
    cos2 = np.cos(theta)**2
    relaxation_superop = pair_relaxation()

    # 4. Calculate the powder spectrum with the supplied relaxation matrix.
    spectrum = simulate_powder(
        spin_system, experiment, options, relaxation_superop,
    )
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


# Relaxation and spin preparation
def relaxation_matrix(transfer_times_s, loss_rates):
    """Combine the native population and coherence blocks in Teacups units."""
    n = len(transfer_times_s)
    rates = (1 / transfer_times_s) / TIME_FACTOR
    diagonal = np.diag_indices(n)
    coherence_block = relaxation.superoperator_coherence_relaxation(
        1 / (T2_NS * 1e-9 * TIME_FACTOR), n,
    )

    rates[diagonal] = -loss_rates / TIME_FACTOR
    population_block = relaxation.superoperator_population_relaxation(rates)
    return population_block + coherence_block


def pair_relaxation():
    """Set transfer times and the same external loss rate for every eigenstate."""
    transfers = np.full((4, 4), T1_US * 1e-6)
    np.fill_diagonal(transfers, np.inf)
    # The two inner states mix S and T0. The T+ <-> T- transfer is absent.
    transfers[1, 2] = transfers[2, 1] = INNER_T1_US * 1e-6
    transfers[0, 3] = transfers[3, 0] = np.inf
    losses = np.full(4, 1 / (PAIR_LIFETIME_US * 1e-6))
    return relaxation_matrix(transfers, losses)


# Teacups solver with a supplied Liouville-space generator
def simulate_powder(spin_system, experiment, options, relaxation_superop):
    """Pass the explicit generator to the Teacups solver for each field block."""
    solve_block = partial(
        powder_field_block,
        system=spin_system,
        experiment=experiment,
        options=options,
        generator=relaxation_superop,
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


def powder_field_block(fields_mt, *, system, experiment, options, generator):
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

        cal.ham_sys = hamiltonians.set_up_rp_hamiltonian(sys, exp, opt, cal)
        hamiltonian = cal.ham_sys + hamiltonians.set_up_mw_hamiltonian(sys, exp, opt, cal)
        vectors = np.linalg.eigh(cal.ham_sys)[1]
        density_matrices.set_up_density_matrix(sys, exp, opt, cal)

        # Build the native commutator with zero default relaxation.
        cal.ham = hamiltonian
        cal.eigvec = vectors
        sys.dynamics = np.zeros((cal.s.dimension, cal.s.dimension), dtype=np.float32)
        hamiltonians.set_up_commutator_superoperator(sys, opt, cal)

        # Supply the population-loss and coherence terms together.
        cal.ham_superop += 1j * relaxation.relaxation_operator_to_hamiltonian_basis(
            generator, vectors,
        )
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
    np.savez("teacups_radical_pair.npz", **result)
    print(f"Calculated {result['spectrum'].shape} map in {result['wall_s']:.3f} s")
