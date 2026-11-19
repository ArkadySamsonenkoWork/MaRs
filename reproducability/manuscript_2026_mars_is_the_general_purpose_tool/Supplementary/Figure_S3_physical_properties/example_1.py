"""Example 1: photoexcited CT and LE triplets and a singlet (block-diagonal composition).

The spin systems and the relaxation are those of the notebook example_article_1.ipynb. and the manuscript_description
The test uses only one single orintation
"""


import sys
import os
import typing as tp
import math

import numpy as np
import torch
import matplotlib.pyplot as plt

#Delite this line if you installed MaRs.
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', '..')))

# Import Mars library components
import mars
from mars import spin_model, spectra_manager, mesher, constants
from mars import utils
from mars import population
from density_evolution import DensityTimeEvolution

# # Let's import some useful functions for plotting simulation results over time.
from mars import visualization

dtype = torch.float64
device = torch.device("cpu")

# We set the experimental parameters here i the beggining:
freq_res = 35 * 1e9  # 35 GHz
temperature = torch.tensor(150.0, dtype=dtype, device=device)

def get_samples(mesh: mars.mesher.BaseMesh) -> tp.Tuple[
    mars.spin_model.SolidSample,
    mars.spin_model.SolidSample,
    mars.spin_model.SolidSample,
    mars.spin_model.SolidSample]:
    """
    Create all spin samples used in the simulation.

    :param mesh : mars.mesher.Mesh
        Orientation mesh over which the sample is defined.
    :return: tuple of the next objects
        singlet : SolidSample
            Singlet (S=0) sample.
        triplet_LE : SolidSample
            Triplet (S=1) sample for the locally‑excited state.
        triplet_CT : SolidSample
            Triplet (S=1) sample for the charge‑transfer state.
        concated_sample : SolidSample
            Block‑diagonal concatenation of CT, LE, and singlet samples.
    """
    
    g_tensor = spin_model.Interaction([2.00, 2.00, 2.00], dtype=dtype, device=device)
    
    # Singlet (S=0) – isotropic g-tensor, energy shift 0.2 eV
    base_spin_system = spin_model.SpinSystem(
        electrons=[0.0],  # S=0 singlet
        g_tensors=[g_tensor],
        energy_shift=constants.unit_converter(0.2, "eV_to_Hz"),
        device=device,
        dtype=dtype,
    )

    singlet = spin_model.SolidSample(
        base_spin_system=base_spin_system,
        ham_strain=0.0,
        gauss=0.00025,
        lorentz=0.00025,
        mesh=mesh,
        device=device,
        dtype=dtype,
    )

    # Triplet LE (S=1) – anisotropic g, ZFS parameters D=2 GHz, E=0.3 GHz. Energy shift is 0.1eV
    g_tensor = spin_model.Interaction([2.00, 2.01, 2.02], dtype=dtype, device=device)
    zfs_interaction = spin_model.DEInteraction([2e9, 0.3e9], dtype=dtype, device=device)  # 2000 and 300 MHz

    base_spin_system = spin_model.SpinSystem(
        electrons=[1.0],  # S=1 triplet
        g_tensors=[g_tensor],
        electron_electron=[(0, 0, zfs_interaction)],
        energy_shift=constants.unit_converter(0.1, "eV_to_Hz"),
        device=device,
        dtype=dtype,
    )

    triplet_le = spin_model.SolidSample(
        base_spin_system=base_spin_system,
        ham_strain=0.0,
        gauss=0.00025,
        lorentz=0.00025,
        mesh=mesh,
        device=device,
        dtype=dtype,
    )
    
    # Triplet CT (S=1) – smaller ZFS D=0.4 GHz, E=0.05 GHz
    g_tensor = spin_model.Interaction([2.00, 2.01, 2.02], dtype=dtype, device=device)
    zfs_interaction = spin_model.DEInteraction([0.4e9, 0.05e9], dtype=dtype, device=device)  # 400 and 50 MHz
    base_spin_system = spin_model.SpinSystem(
        electrons=[1.0],  # S=1 triplet
        g_tensors=[g_tensor],
        electron_electron=[(0, 0, zfs_interaction)],
        device=device,
        dtype=dtype
    )
    
    triplet_ct = spin_model.SolidSample(
        base_spin_system=base_spin_system,
        ham_strain=0.0,
        gauss=0.00025,
        lorentz=0.00025,
        mesh=mesh,
        device=device,
        dtype=dtype,
    )

    # Concatenate CT, LE, singlet (block‑diagonal)
    concated_sample = mars.concat((triplet_ct, triplet_le, singlet))
    return singlet, triplet_le, triplet_ct, concated_sample

def get_decay_contexts(singlet, triplet_le, triplet_ct):
    """
    Create decay contexts for singlet and triplet states.

    :param singlet: Singlet sample.
    :param triplet_le: LE triplet sample.
    :param triplet_ct: CT triplet sample.
    :return: Singlet decay (rate 1e7 s^-1),
    LE triplet decay (rate 1e3 s^-1 per sublevel, xyz basis), CT triplet decay (rate 1e3 s^-1 per sublevel, xyz basis).
    """
    # We do not nessusary need to set basis here, since it has only one state
    ctx_singlet = population.Context(init_populations=[1.0], decay_rates=[1e7], device=device, dtype=dtype) # We set the decay in 1e7 s^-1
    
    # Herein we already need basis "xyz" and samples which set this basis
    ctx_le_decay = population.Context(basis="xyz", sample=triplet_le, decay_rates=[1e3, 1e3, 1e3],
                                      device=device, dtype=dtype)
    ctx_ct_decay = population.Context(basis="xyz", sample=triplet_ct, decay_rates=[1e3, 1e3, 1e3],
                                      device=device, dtype=dtype)
    
    return ctx_singlet, ctx_le_decay, ctx_ct_decay

class SpectralDensity:
    """
    Calculates the Spectral Density function based on a set of mode frequencies 
    and temperature, utilizing a Lorentzian broadening model.
    
    Spectral density for a set of harmonic modes with Lorentzian broadening.

    For each mode k with frequency ω_k (in cm^-1), the spectral density at probe
    frequency ω (rad/s) and temperature T is:
        J(ω) = Σ_k [ δ/(δ^2 + (ω_k+ω)^2) * n(ω_k) + δ/(δ^2 + (ω_k-ω)^2) * (n(ω_k)+1) ]
    where n(ω_k) = 1/(exp(ħω_k/k_B T) - 1) and δ = sqrt(ω_k^2 * exp(ħω_k/k_B T) * n^2)
    (the broadening is chosen to preserve detailed balance).
    
    """

    def __init__(self, w_k: tp.Union[float, torch.Tensor],
                 dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")):
        """
        Initialize the SpectralDensity calculator.

        :param w_k: Mode frequencies in inverse centimeters (cm^-1).
        :param dtype: Data type for floating point operations.
        :param device: Computation device.
        """
        super().__init__()
        w_k_hz = constants.unit_converter(w_k, "cm-1_to_Hz")
        self.w_k_rad = torch.tensor(w_k_hz, dtype=dtype, device=device) * (2 * math.pi)
        
        # Store the temperature-independent energy conversion for use in __call__
        self.w_k_K = torch.tensor(constants.unit_converter(w_k, "cm-1_to_K"), dtype=dtype, device=device)

    def __call__(self, w_rad: tp.Union[torch.Tensor, float], temperature: tp.Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the spectral density at given probe frequencies and temperature.

        :param w_rad: Probe frequencies in radians per second (rad/s).
        :param temperature: System temperature in Kelvin.
        :return: Spectral density values calculated at the probe frequencies.
        """
        # Calculate temperature-dependent terms dynamically
        x = self.w_k_K / temperature
        exp_val = torch.exp(x)
        occup = 1.0 / torch.expm1(x)
        
        # Calculate the broadening parameter (delta)
        delta = torch.sqrt(self.w_k_rad ** 2 * exp_val * occup ** 2)

        plus = self.w_k_rad + w_rad
        minus = self.w_k_rad - w_rad
        
        term_plus = self._lorentz(plus, delta) * occup
        term_minus = self._lorentz(minus, delta) * (occup + 1.0)
        
        return term_plus + term_minus

    def _lorentz(self, arg: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Compute the Lorentzian lineshape function.

        :param arg: Frequency argument (typically sum or difference of frequencies).
        :param delta: Broadening parameter (depends on temperature).
        :return: Lorentzian value delta / (delta^2 + arg^2).
        """
        return delta / (delta ** 2 + arg ** 2)

    def __repr__(self) -> str:
        """
        Return a string representation of the object.

        :return: String containing class name and shape of mode frequencies.
        """
        return f"SpectralDensity(modes={self.w_k_rad.shape}, device={self.w_k_rad.device})"

def get_copling_operators(triplet):
    """
    Extract coupling operators for librations along the x, y, z axes.

    For each axis, we obtain the static and field‑dependent operators (in Hz)
    and convert them to rad/s.

    :param triplet: Triplet sample for which to compute the operators.
    :return: List of tuples, each containing (O_static_rad, O_field_rad) for one axis.
    """
    coupling_operators = []
    for i in range(3):
        n = torch.zeros((3, ), dtype=dtype, device=device)
        n[i] = 1.0
        O_static, O_field = triplet.get_librations_along_axis(n)
        O_static = O_static * (2 * math.pi)
        O_field = O_field * (2 * math.pi)
        coupling_operators.append(
            (O_static, O_field)
        )
    return coupling_operators

def get_context_triplet_intra(freq, triplet):
    """
    Build a context for intra‑triplet Redfield relaxation.

    :param freq: Mode frequencies in cm⁻¹ for each axis.
    :param triplet: Triplet sample.
    :return: Relaxation context for intra‑triplet processes.
    """
    coupling_operators = get_copling_operators(triplet)
    channels = []
    
    for i, freq in enumerate(frequencies):
        spectral_density_func = SpectralDensity(freq, dtype, device)
        channels.append(
            population.RedfieldRelaxationChannel(operator_components=[coupling_operators[i]],
                                                 spectral_density_func=spectral_density_func,
                                                 thermal_balance_mode="skip")  # The thermal balans is already set in spectral function
        )

    context_triplet_intra = population.Context(
        basis="product", sample=triplet,
        relaxation_coupling_channels=channels, device=device, dtype=dtype
    )
    return context_triplet_intra

frequencies = [100, 100, 100]

def get_triplet_singlet_rates(
    vector_xyz_ct: tp.Union[list, tuple, torch.Tensor],
    vector_xyz_le: tp.Union[list, tuple, torch.Tensor],
    triplet_ct,
    triplet_le,
    full_sample,
) -> torch.Tensor:
    """
    Construct a 7x7 symmetric relaxation rate matrix connecting triplet-CT sublevels,
    triplet-LE sublevels, and the singlet state in the zero-field splitting (ZFS) basis.
    
    This function maps the provided relaxation rates for the CT and LE triplet sublevels 
    to their corresponding transitions towards the Singlet state (S1) and between each other,
    within a 7-level system. The ZFS basis may have a different ordering than the xyz triplet 
    basis, so this function matches eigenvectors between the two bases to ensure correct 
    rate assignment.
    
    Assumed energy ordering: ³CT < ³LE < S₁
    
    :param vector_xyz_ct: list | tuple | torch.Tensor
        A vector of length 3 containing relaxation rates [k_x, k_y, k_z] for transitions 
        between ³CT sublevels (Tx, Ty, Tz) and the Singlet state (S1). 
        Order: [S -> CTx, S -> CTy, S -> CTz]. Values should be non-negative.
    
    :param vector_xyz_le: list | tuple | torch.Tensor
        A vector of length 3 containing relaxation rates [k_x, k_y, k_z] for transitions 
        between ³LE sublevels (Tx, Ty, Tz) and the Singlet state (S1). 
        Order: [S -> LEx, S -> LEy, S -> LEz]. Values should be non-negative.

    :param triplet_ct: object
        Object representing the triplet-CT manifold. Must implement `.get_xyz_basis()[0]` 
        returning a (3, 3) tensor of xyz basis vectors for the CT triplet.
    
    :param triplet_le: object
        Object representing the triplet-LE manifold. Must implement `.get_xyz_basis()[0]` 
        returning a (3, 3) tensor of xyz basis vectors for the LE triplet.
    
    :param full_sample: object
        Object representing the full 7-level system. Must implement 
        `.get_zero_field_splitting_basis()[0]` returning a (7, 7) tensor of ZFS eigenvectors.
        Assumes columns 0-5 are triplet states and column 6 is the singlet state.
    
    :return: torch.Tensor
        A 7x7 symmetric matrix (torch.Tensor) representing the relaxation rates in the ZFS basis.
        Indices 0-5 correspond to the 6 matched triplet sublevels (3 CT + 3 LE).
        Index 6 corresponds to the Singlet state (S1).
        The matrix is symmetric: rate[i, j] = rate[j, i].
    """
    # 1. Convert inputs to tensors & set device/dtype
    rates_ct = torch.as_tensor(vector_xyz_ct, dtype=torch.float64)
    rates_le = torch.as_tensor(vector_xyz_le, dtype=torch.float64)
    device = rates_ct.device
    
    # 2. Extract local xyz bases (3, 3) and embed into global 7D space
    xyz_ct = triplet_ct.get_xyz_basis()[0].to(device=device, dtype=torch.complex128)
    xyz_le = triplet_le.get_xyz_basis()[0].to(device=device, dtype=torch.complex128)
    
    # Embed: CT -> indices 0-2, LE -> indices 3-5
    xyz_global = torch.zeros((7, 6), dtype=torch.complex128, device=device)
    xyz_global[:3, :3] = xyz_ct
    xyz_global[3:6, 3:] = xyz_le
    
    # 3. Extract ZFS triplet subspace from the global (7, 7) basis
    zfs_basis = full_sample.get_zero_field_splitting_basis()[0].to(device=device, dtype=torch.complex128)
    zfs_triplets = zfs_basis[:, :6]  # shape (7, 6)
    
    # 4. Compute overlap matrix |<xyz_i | zfs_j>|^2 (shape: 6 x 6)
    overlap_matrix = torch.abs(torch.matmul(xyz_global.conj().T, zfs_triplets)) ** 2
    
    # 5. Greedy 1-to-1 mapping from xyz indices to ZFS indices
    mapping = torch.full((6,), -1, dtype=torch.long, device=device)
    overlap_copy = overlap_matrix.clone()
    
    for _ in range(6):
        idx_flat = torch.argmax(overlap_copy)
        idx_xyz = idx_flat // 6
        idx_zfs = idx_flat % 6
        
        mapping[idx_xyz] = idx_zfs
        overlap_copy[:, idx_zfs] = -1.0  # Mask used ZFS column
        overlap_copy[idx_xyz, :] = -1.0  # Mask used XYZ row
        
    ct_zfs_indices = mapping[:3]
    le_zfs_indices = mapping[3:]
    singlet_idx = 6
    
    rate_matrix = torch.zeros((7, 7), dtype=rates_ct.dtype, device=device)
    
    # CT <-> S transitions
    for i in range(3):
        rate_matrix[ct_zfs_indices[i], singlet_idx] = rates_ct[i]
        rate_matrix[singlet_idx, ct_zfs_indices[i]] = rates_ct[i]
        
    # LE <-> S transitions
    for i in range(3):
        rate_matrix[le_zfs_indices[i], singlet_idx] = rates_le[i]
        rate_matrix[singlet_idx, le_zfs_indices[i]] = rates_le[i]
    return rate_matrix

def get_triplet_triplet_rates(
    matrix_ct_le: tp.Union[list, tuple, torch.Tensor],
    full_sample,
) -> torch.Tensor:
    """
    Build a 7×7 rate matrix for CT↔LE triplet transitions (off‑diagonal block).

    :param matrix_ct_le: Rates from CT sublevels (rows) to LE sublevels (columns) in the ZFS basis.
    :param full_sample: Full concatenated sample.
    :return: 7×7 rate matrix with the CT‑LE block filled.
    """
    
    ct_idx = torch.tensor([0, 1, 2], device=device)
    le_idx = torch.tensor([3, 4, 5], device=device)
    rates_ct_le = torch.as_tensor(matrix_ct_le, dtype=torch.float64)
    
    tot_rates = torch.zeros((7, 7), dtype=dtype, device=device)
    
    tot_rates[0:3, 3:6] = rates_ct_le
    tot_rates[3:6, 0:3] = rates_ct_le.T
    return tot_rates

def get_exchange_contexts(singlet, triplet_ct, triplet_le, full_sample):
    """
    Create contexts for singlet–triplet and triplet–triplet exchange.

    :param singlet: Singlet sample.
    :param triplet_ct: CT triplet sample.
    :param triplet_le: LE triplet sample.
    :param full_sample: Full concatenated sample.
    :return: Singlet–triplet exchange (defined in ZFS basis), CT–LE triplet exchange (defined in eigenbasis).
    """
    rate_matrix = get_triplet_singlet_rates(
        vector_xyz_ct=[1e5, 1e5, 2e5],
        vector_xyz_le=[1e7, 1e7, 2e7],
        triplet_ct=triplet_ct,
        triplet_le=triplet_le,
        full_sample=full_sample
    )

    exch_singlet = population.Context(
        basis="zfs", sample=full_sample,
        thermal_rates=rate_matrix, device=device, dtype=dtype
    )

    thermal_rates = get_triplet_triplet_rates(
        matrix_ct_le = [[1e6, 0.0, 0.0], 
                      [0.0, 1e6, 0.0], 
                      [0.0, 0.0, 1e6]],
        full_sample=full_sample)

    exch_triplet = population.Context(
        basis="eigen", sample=full_sample,
        thermal_rates=thermal_rates, device=device, dtype=dtype
    )
    return exch_singlet, exch_triplet

def build_example(secular=False, coherent=True, initial_state="context", initial_grid_frequency=5,
                  fields=(1.15, 1.35), t_max=5e-06, n_times=60):
    """Example 1 for one crystal orientation, prepared for DensityTimeEvolution.

    :param secular: Use the secular part of the relaxation superoperator.
    :param coherent: Include the Hamiltonian and the microwave field (False: relaxation superoperator alone).
    :param initial_state: "context" (initial density of the context) or "basis" (all |i><j|).
    
    :param fields: Field range (T) in which the resonance fields are searched.
    :param t_max: Largest time (s).
    :param n_times: Number of time points (logarithmic grid).
    :return: dict with the sample, the creator (DensityTimeEvolution), the fields, the times and a title.
    """
    mesh = mesher.DelaunayMesh(initial_grid_frequency=initial_grid_frequency, dtype=dtype, device=device)
    singlet, triplet_le, triplet_ct, full_sample = get_samples(mesh)
    ctx_singlet, ctx_le_decay, ctx_ct_decay = get_decay_contexts(singlet, triplet_le, triplet_ct)

    ctx_le_intra = get_context_triplet_intra(frequencies, triplet_le)
    ctx_ct_intra = get_context_triplet_intra(frequencies, triplet_ct)

    exch_singlet, exch_triplet = get_exchange_contexts(singlet, triplet_ct, triplet_le, full_sample)
    ctx_le_total = ctx_le_intra + ctx_le_decay
    ctx_ct_total = ctx_ct_intra + ctx_ct_decay
    block_ctx = mars.concat((ctx_ct_total, ctx_le_total, ctx_singlet))
    final_relax = block_ctx + exch_singlet + exch_triplet


    creator = DensityTimeEvolution(freq=freq_res, sample=full_sample, context=final_relax, temperature=temperature,
                                   secular=secular, coherent=coherent, initial_state=initial_state,
                                   device=device, dtype=dtype)
    tau = 1 / freq_res
    time = torch.logspace(math.log10(10 * tau), math.log10(t_max), n_times, dtype=dtype, device=device)
    return dict(sample=full_sample, creator=creator,
                fields=torch.linspace(fields[0], fields[1], 1000, dtype=dtype, device=device),
                time=time, title=f"Example 1: CT and LE triplets + singlet")
