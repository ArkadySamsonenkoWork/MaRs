"""Example 2: exchange-coupled doublet and triplet (tensor-product composition).

The spin systems and the relaxation are those of the notebook example_article_2.ipynb. and the manuscript
The only change is the orientation mesh, which is passed to get_samples.
"""


# Import necessary libraries
import sys
import os
import math

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..', "..", "..")))

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

# Experimental parameters
freq_res = 9.8 * 1e9          # 9.8 GHz microwave frequency
temperature = torch.tensor(100.0, dtype=dtype, device=device)   # Temperature in Kelvin

# Let's start from the sample creation
def get_samples(exchange_value: float, mesh=None):
    """
    Create the doublet, triplet, and coupled complex samples.

    :param exchange_value: Exchange coupling constant (in Hz) between doublet and triplet.
    :param mesh: Orientation mesh of the samples (None: the default powder mesh).
    :return: Doublet (S=1/2) sample with anisotropic g-tensor, Triplet (S=1) sample with ZFS (D=500 MHz, E=100 MHz), Coupled doublet–triplet system with the given exchange interaction.
    """
    g_tensor_triplet = spin_model.Interaction([2.000, 2.002, 2.004], dtype=dtype, device=device)
    zfs_interaction = spin_model.DEInteraction([500e6, 100e6], dtype=dtype, device=device)  # 500 and 100 MHz

    base_spin_system = spin_model.SpinSystem(
        electrons=[1.0],  # S=1 triplet
        g_tensors=[g_tensor_triplet],
        electron_electron=[(0, 0, zfs_interaction)]
    )

    triplet = spin_model.SolidSample(
        base_spin_system=base_spin_system,
        ham_strain=0.0,
        mesh=mesh,
        gauss=0.002,
        lorentz=0.0015,
        device=device,
        dtype=dtype
    )
    
    # Doublet (S=1/2) – anisotropic g
    g_tensor_doublet = spin_model.Interaction([2.02, 2.04, 2.25], dtype=dtype, device=device)
    base_spin_system = spin_model.SpinSystem(
        electrons=[0.5],  # S=0.5
        g_tensors=[g_tensor_doublet],
    )
    doublet = spin_model.SolidSample(
        base_spin_system=base_spin_system,
        ham_strain=0.0,
        mesh=mesh,
        gauss=0.002,
        lorentz=0.002,
        device=device,
        dtype=dtype
    )
    
    # Complex coupled system: doublet + triplet with exchange
    exchange_interaction = spin_model.Interaction(exchange_value, dtype=dtype, device=device)   # Here we define exchange interaction as JS1S2. In the article this is -2JS1S2
    base_spin_system = spin_model.SpinSystem(
        electrons=[0.5, 1.0],  # S=0.5 triplet
        g_tensors=[g_tensor_doublet, g_tensor_triplet],
        electron_electron=[(1, 1, zfs_interaction), (0, 1, exchange_interaction)],
    )

    sample_complex = spin_model.SolidSample(
        base_spin_system=base_spin_system,
        ham_strain=0.0,
        mesh=mesh,
        gauss=0.002,
        lorentz=0.002,
        device=device,
        dtype=dtype
    )
    return doublet, triplet, sample_complex

def get_contexts(temperature, doublet, triplet, sample_complex):
    """
    Build relaxation contexts for the doublet, triplet, and coupled system.

    :param temperature: Temperature used for Boltzmann populations.
    :param doublet: Doublet sample.
    :param triplet: Triplet sample.
    :param sample_complex: Coupled sample.
    :return: Doublet context with thermal rates,
    Triplet context with initial population and thermal rates,
    Coupled context (product of triplet and doublet) with dephasing.
    """
    
    # Compute Boltzmann populations for doublet
    fields = torch.tensor([0.34], dtype=dtype)
    F, _, _, Gz = doublet.get_hamiltonian_terms()
    H = F + Gz * fields
    H  = H.unsqueeze(-3)
    energies, full_system_vectors = torch.linalg.eigh(H)

    populations_doublet = torch.softmax(
        constants.unit_converter(-energies, "Hz_to_K") / temperature, dim=-1
                                )[0, 0]
    
    # Triplet context in xyz basis with initial population in the third sublevel
    context_triplet_xyz = population.Context(
        basis="xyz",
        sample=triplet,
        init_populations=[0.0, 0.0, 1.0],
        decay_rates=[0.0, 0.0, 0.0],
        dtype=dtype,
    )

    # Triplet thermalisation among Zeeman sublevels
    context_triplet_zeeman = population.Context(
        basis="zeeman",
        sample=triplet,
        thermal_rates=torch.tensor([[-0.0, 1e3, 0.0],
                                     [1e3, -0.0, 1e3],
                                     [0.0, 1e3, -0.0]], dtype=dtype),
        dtype=dtype,
    )

    # Doublet context with initial thermal populations and thermal rates
    context_doublet = population.Context(
        basis="zeeman",
        sample=doublet,
        init_populations=populations_doublet,
        thermal_rates=torch.tensor([[-0.0, 1e4],
                                     [1e4, -0.0]], dtype=dtype),
        dtype=dtype
    )

    # Dephasing for the coupled system
    context_dephasing = population.Context(
        basis="eigen",
        sample=sample_complex,
        dephasing=[1e6 for _ in range(6)],
        dtype=dtype
    )

    # Coupled context: tensor product of triplet (xyz + zeeman) and doublet contexts
    context_triplet_total = context_triplet_xyz + context_triplet_zeeman
    context_complex = mars.multiply((context_doublet, context_triplet_total)) + context_dephasing

    return context_doublet, context_triplet_total, context_complex

def build_example(secular=False, coherent=True, initial_state="context", initial_grid_frequency=5,
                  exchange=300e6, fields=(0.28, 0.42), t_max=0.0001, n_times=60):
    """Example 2 for one crystal orientation, prepared for DensityTimeEvolution.

    :param secular: Use the secular part of the relaxation superoperator.
    :param coherent: Include the Hamiltonian and the microwave field (False: relaxation superoperator alone).
    :param initial_state: "context" (initial density of the context) or "basis" (all |i><j|).

    :param exchange: Exchange constant (Hz).
    :param fields: Field range (T) in which the resonance fields are searched.
    :param t_max: Largest time (s).
    :param n_times: Number of time points (logarithmic grid).
    :return: dict with the sample, the creator (DensityTimeEvolution), the fields, the times and a title.
    """
    mesh = mesher.DelaunayMesh(initial_grid_frequency=initial_grid_frequency, dtype=dtype, device=device)
    doublet, triplet, sample_complex = get_samples(exchange, mesh)
    context_doublet, context_triplet, context_complex = get_contexts(temperature, doublet, triplet, sample_complex)

    creator = DensityTimeEvolution(freq=freq_res, sample=sample_complex, context=context_complex, temperature=temperature,
                                   secular=secular, coherent=coherent, initial_state=initial_state,
                                   device=device, dtype=dtype)
    tau = 1 / freq_res
    time = torch.logspace(math.log10(10 * tau), math.log10(t_max), n_times, dtype=dtype, device=device)
    return dict(sample=sample_complex, creator=creator,
                fields=torch.linspace(fields[0], fields[1], 1000, dtype=dtype, device=device),
                time=time, title=f"Example 2: doublet–triplet complex, J = {exchange / 1e6:g} MHz")
