import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, root_dir)

import time
import torch
import typing as tp
from mars import spin_model, mesher, constants


def create_2_electrons_sample(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with two coupled S=1/2 electrons.

    This configuration models a simple diradical system with exchange and
    dipolar coupling between electrons.

    Parameters
    ----------
    mesh : BaseMesh or tuple of (int, int), optional
    device : torch.device, optional
        Computation device (cpu or cuda). Default is cpu.
    dtype : torch.dtype, optional
        Floating point precision.

    Returns
    -------
    SolidSample
        Configured sample ready for spectrum simulation.
    """

    g_tensor = spin_model.Interaction(
        (2.02, 2.04, 2.06),
        device=device,
        dtype=dtype
    )

    exchange = spin_model.Interaction(
        constants.unit_converter(1.0, "cm-1_to_Hz"),
        device=device,
        dtype=dtype
    )

    dipolar = spin_model.DEInteraction(
        [100e6, 10e6],  # D=100 MHz, E=10 MHz
        device=device,
        dtype=dtype
    )

    total_interaction = exchange + dipolar

    # Build spin system
    spin_sys = spin_model.SpinSystem(
        electrons=[0.5, 0.5],
        g_tensors=[g_tensor, g_tensor],
        electron_electron=[(0, 1, total_interaction)],
        device=device,
        dtype=dtype
    )

    # Create sample with realistic broadening parameters
    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        ham_strain=5e7,  # 50 MHz Hamiltonian strain
        gauss=0.001,  # 1 mT Gaussian broadening
        lorentz=0.001,  # 1 mT Lorentzian broadening
        mesh=mesh,
        device=device,
        dtype=dtype
    )

    return sample


def create_3_electrons_sample(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with three coupled S=1/2 electrons.

    Models a triradical system with pairwise exchange/dipolar couplings.

    Parameters
    ----------
    mesh : BaseMesh or tuple of (int, int), optional
        Orientation mesh for powder averaging.
    device : torch.device, optional
        Computation device. Default is cpu.
    dtype : torch.dtype, optional
        Floating point precision. Default is float64.

    Returns
    -------
    SolidSample
    """
    g_tensor = spin_model.Interaction((2.00, 2.00, 2.00), device=device, dtype=dtype)

    exchange_val = constants.unit_converter(0.5, "cm-1_to_Hz")
    exchange = spin_model.Interaction(exchange_val, device=device, dtype=dtype)

    dipolar12 = spin_model.DEInteraction([80e6, 8e6], device=device, dtype=dtype)
    dipolar13 = spin_model.DEInteraction([60e6, 6e6], device=device, dtype=dtype)
    dipolar23 = spin_model.DEInteraction([70e6, 7e6], device=device, dtype=dtype)

    spin_sys = spin_model.SpinSystem(
        electrons=[0.5, 0.5, 0.5],
        g_tensors=[g_tensor, g_tensor, g_tensor],
        electron_electron=[
            (0, 1, exchange + dipolar12),
            (0, 2, exchange + dipolar13),
            (1, 2, exchange + dipolar23)
        ],
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        ham_strain=3e7,
        gauss=0.0015,
        lorentz=0.0015,
        mesh=mesh,
        device=device,
        dtype=dtype
    )
    return sample


def create_4_electrons_sample(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with four coupled S=1/2 electrons.

    Models a tetraradical system. Hilbert space dimension = 16 states.

    Parameters
    ----------
    mesh : BaseMesh or tuple of (int, int), optional
        Orientation mesh. Use coarse mesh (e.g., (30,30)) for initial tests.
    device : torch.device, optional
        Computation device. GPU strongly recommended.
    dtype : torch.dtype, optional
        Floating point precision. float32 acceptable for timing tests.

    Returns
    -------
    SolidSample
    """
    g_tensor = spin_model.Interaction((2.00, 2.00, 2.00), device=device, dtype=dtype)
    exchange_val = constants.unit_converter(0.3, "cm-1_to_Hz")
    exchange = spin_model.Interaction(exchange_val, device=device, dtype=dtype)

    interactions = []
    dipolar_vals = [(100e6, 10e6), (100e6, 10e6), (100e6, 10e6), (100e6, 10e6), (100e6, 10e6), (100e6, 10e6)]

    pair_idx = 0
    for i in range(4):
        for j in range(i + 1, 4):
            dipolar = spin_model.DEInteraction(dipolar_vals[pair_idx], device=device, dtype=dtype)
            interactions.append((i, j, exchange + dipolar))
            pair_idx += 1

    spin_sys = spin_model.SpinSystem(
        electrons=[0.5, 0.5, 0.5, 0.5],
        g_tensors=[g_tensor] * 4,
        electron_electron=interactions,
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        ham_strain=2e7,
        gauss=0.002,
        lorentz=0.002,
        mesh=mesh,
        device=device,
        dtype=dtype
    )
    return sample


def create_2_electrons_1_nuclei_sample(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with two electrons coupled to a single nitrogen nucleus.

    Models a system where both electrons share hyperfine coupling to one ¹⁴N
    nucleus (I=1). Hilbert space dimension = 12 states

    Parameters
    ----------
    mesh : BaseMesh or tuple of (int, int), optional
        Orientation mesh for powder averaging.
    device : torch.device, optional
        Computation device.
    dtype : torch.dtype, optional
        Floating point precision.

    Returns
    -------
    SolidSample

    """
    g_tensor = spin_model.Interaction((2.02, 2.04, 2.06), device=device, dtype=dtype)
    A_tensor = spin_model.Interaction((20e6, 20e6, 70e6), device=device, dtype=dtype)  # MHz

    exchange = spin_model.Interaction(
        constants.unit_converter(0.8, "cm-1_to_Hz"),
        device=device,
        dtype=dtype
    )
    dipolar = spin_model.DEInteraction([100e6, 10e6], device=device, dtype=dtype)

    spin_sys = spin_model.SpinSystem(
        electrons=[0.5, 0.5],
        nuclei=["14N"],
        g_tensors=[g_tensor, g_tensor],
        electron_nuclei=[
            (0, 0, A_tensor),  # e0 coupled to N
            (1, 0, A_tensor)  # e1 coupled to same N
        ],
        electron_electron=[(0, 1, exchange + dipolar)],
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        ham_strain=4e7,
        gauss=0.0012,
        lorentz=0.0012,
        mesh=mesh,
        device=device,
        dtype=dtype
    )
    return sample


def create_1_high_spin_electron_1_nuclei_sample(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with one high-spin electron (S=5/2) coupled to ⁵⁵Mn nucleus.

    Models Mn(II) ion system with zero-field splitting and hyperfine coupling.
    Hilbert space dimension = 36 states. Computationally intensive due to
    large matrix diagonalization requirements.

    Parameters
    ----------
    mesh : BaseMesh or tuple of (int, int), optional
        Orientation mesh. Coarse mesh recommended for initial timing tests.
    device : torch.device, optional
        Computation device. GPU essential for reasonable performance.
    dtype : torch.dtype, optional
        Floating point precision. float64 required for ZFS accuracy.

    Returns
    -------
    SolidSample

    """
    # Anisotropic g-tensor for Mn(II)
    g_tensor = spin_model.Interaction((2.02, 2.04, 2.10), device=device, dtype=dtype)

    # Isotropic ⁵⁵Mn hyperfine (I=5/2)
    A_hf = spin_model.Interaction(300e6, device=device, dtype=dtype)  # 300 MHz

    zfs = spin_model.DEInteraction(
        [500e6, 100e6],
        strain=[50e6, 10e6],
        device=device,
        dtype=dtype
    )

    spin_sys = spin_model.SpinSystem(
        electrons=[5 / 2],  # S = 5/2
        nuclei=["55Mn"],  # I = 5/2
        g_tensors=[g_tensor],
        electron_nuclei=[(0, 0, A_hf)],
        electron_electron=[(0, 0, zfs)],
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        ham_strain=6e7,
        gauss=0.0015,
        lorentz=0.0015,
        mesh=mesh,
        device=device,
        dtype=dtype
    )

    return sample


def create_2_middle_spin_sample(
        mesh: tp.Optional[tp.Union[mesher.BaseMesh, tp.Tuple[int, int]]] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64
) -> spin_model.SolidSample:
    """
    Create a sample with one high-spin electron (S=5/2) coupled to ⁵⁵Mn nucleus.

    Models Mn(II) ion system with zero-field splitting and hyperfine coupling.
    Hilbert space dimension = 36 states. Computationally intensive due to
    large matrix diagonalization requirements.

    Parameters
    ----------
    mesh : BaseMesh or tuple of (int, int), optional
        Orientation mesh. Coarse mesh recommended for initial timing tests.
    device : torch.device, optional
        Computation device. GPU essential for reasonable performance.
    dtype : torch.dtype, optional
        Floating point precision. float64 required for ZFS accuracy.

    Returns
    -------
    SolidSample

    """
    g_tensor = spin_model.Interaction((2.02, 2.04, 2.12), device=device, dtype=dtype)

    zfs = spin_model.DEInteraction(
        [500e6, 100e6],
        device=device,
        dtype=dtype
    )

    dipolar = spin_model.DEInteraction(
        [100e6, 10e6],
        device=device,
        dtype=dtype
    )

    spin_sys = spin_model.SpinSystem(
        electrons=[3/2, 3/2],
        g_tensors=[g_tensor, g_tensor],
        electron_electron=[(0, 0, zfs), (0, 0, zfs), (0, 1, dipolar)],
        device=device,
        dtype=dtype
    )

    sample = spin_model.SolidSample(
        base_spin_system=spin_sys,
        ham_strain=6e7,
        gauss=0.0015,
        lorentz=0.0015,
        mesh=mesh,
        device=device,
        dtype=dtype
    )
    return sample